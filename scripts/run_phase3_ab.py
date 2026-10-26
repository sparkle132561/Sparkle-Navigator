"""阶段三 A/B 对比实验：对冻结数据集按变体运行真实模型并产出报告。

针对免费档限流设计：请求间隔节流 + 429 指数退避重试 + 断点续跑
（已 COMPLETED 的 case 跳过），可长时间后台运行。

用法（需已配置 Tau Provider 凭据，真实调用会产生费用）：

    python scripts/run_phase3_ab.py --variants A,B
    python scripts/run_phase3_ab.py --variants A --limit 2   # 试跑
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sparkle_navigator.domain.evaluation import (
    AgentRunStatus,
    EvaluationCase,
    EvaluationRequest,
)
from sparkle_navigator.evaluation.core import EvaluationLab
from sparkle_navigator.evaluation.native_backend import NativeBackend
from sparkle_navigator.evaluation.phase3_experiment import (
    Phase3Evaluator,
    build_variant_session_config,
    load_phase3_dataset,
    variant_frozen_summary,
    verify_evidence_anchors,
)
from sparkle_navigator.presentation.tui.services import (
    SparklePresentationServices,
    default_sparkle_paths,
)
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.session import JsonlSessionStorage
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding import CodingSessionConfig
from tau_coding.credentials import FileCredentialStore
from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    load_provider_settings,
    openai_compatible_config_from_provider,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

LOCKED_SOURCE_COMMIT = "a8f18e7bc645d09d47d40b21bc602df97f2579fd"
DATASET_DIR = REPO_ROOT / "data" / "benchmarks" / "phase3-ab"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_REQUESTS_PER_RUN = 30
DEFAULT_RUNS_CAP = 40
DEFAULT_MIN_INTERVAL_SECONDS = 45.0
DEFAULT_BACKOFF_START_SECONDS = 60.0
DEFAULT_BACKOFF_MAX_SECONDS = 300.0
DEFAULT_RATE_LIMIT_ATTEMPTS = 8
PROVIDER_TIMEOUT_SECONDS = 90.0
PROVIDER_MAX_RETRIES = 0


class ProviderRequestLimitExceeded(RuntimeError):
    """当前 AgentRun 已达到成功请求数上限。"""


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "429" in message or "rate limit" in message


class PatientRateLimitedProvider(ModelProvider):
    """请求间隔节流、429 指数退避重试、成功请求数上限。

    只有在尚未向调用方产出任何事件时才重试（失败请求未消耗生成）；
    已开始的流式中途失败原样上抛。计数只统计成功完成的请求。
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_requests: int,
        min_interval_seconds: float,
        backoff_start_seconds: float,
        backoff_max_seconds: float,
        max_attempts: int,
    ) -> None:
        self._inner = inner
        self._max_requests = max_requests
        self._min_interval = min_interval_seconds
        self._backoff_start = backoff_start_seconds
        self._backoff_max = backoff_max_seconds
        self._max_attempts = max_attempts
        self._count = 0
        self._last_start = 0.0

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list,
        tools: list,
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> Any:
        async def stream() -> Any:
            delay = self._backoff_start
            attempts = 0
            while True:
                wait = self._min_interval - (time.monotonic() - self._last_start)
                if wait > 0:
                    await asyncio.sleep(wait)
                if self._count >= self._max_requests:
                    raise ProviderRequestLimitExceeded(
                        "AgentRun 已达到成功请求数上限。"
                    )
                self._last_start = time.monotonic()
                produced = False
                try:
                    async for event in self._inner.stream_response(
                        model=model,
                        system=system,
                        messages=messages,
                        tools=tools,
                        signal=signal,
                        session_id=session_id,
                    ):
                        produced = True
                        yield event
                    self._count += 1
                    return
                except Exception as exc:  # noqa: BLE001
                    if produced or not _is_rate_limit_error(exc):
                        raise
                    attempts += 1
                    if attempts >= self._max_attempts:
                        raise
                    print(
                        f"    [rate-limit] 第 {attempts} 次限流，等待 {delay:.0f}s 后重试",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._backoff_max)

        return stream()

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="A,B", help="逗号分隔变体，如 A,B")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 Case（试跑）")
    parser.add_argument(
        "--cases", default=None, help="逗号分隔 case_id 过滤（断点补跑指定题）"
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--requests-per-run", type=int, default=DEFAULT_REQUESTS_PER_RUN
    )
    parser.add_argument("--runs-cap", type=int, default=DEFAULT_RUNS_CAP)
    parser.add_argument(
        "--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_SECONDS
    )
    parser.add_argument(
        "--outdir", default=None, help="输出目录（默认 .scratch/phase3-ab）"
    )
    return parser.parse_args()


def resolve_tau_repository(
    services: SparklePresentationServices,
) -> tuple[str, str, Path]:
    """返回 (repository_id, snapshot_id, 工作副本目录)，校验锁定 commit。"""

    repositories = asyncio.run(services.list_repositories())
    tau = next((r for r in repositories if r.name == "tau"), None)
    if tau is None or not tau.current_snapshot_id:
        raise SystemExit("本地数据库中没有名为 tau 的 Repository（含 Snapshot）。")
    snapshot = asyncio.run(
        services.get_repository_snapshot(tau.repository_id, tau.current_snapshot_id)
    )
    if snapshot.source_commit != LOCKED_SOURCE_COMMIT:
        raise SystemExit(
            f"tau 快照 commit {snapshot.source_commit[:8]} 与锁定的 "
            f"{LOCKED_SOURCE_COMMIT[:8]} 不一致，拒绝运行。"
        )
    working_copy = asyncio.run(services.open_working_copy(tau.repository_id))
    if working_copy.snapshot_id != tau.current_snapshot_id:
        raise SystemExit("工作副本 Snapshot 与当前 Snapshot 不一致，请重新导入。")
    return tau.repository_id, tau.current_snapshot_id, working_copy.root


def build_live_factory(
    *,
    variant: str,
    runtime_provider_config: Any,
    provider_name: str,
    model: str,
    database_path: Path,
    snapshot_id: str,
    repository_id: str,
    context_database_path: Path,
    cwd: Path,
    max_tokens: int,
    requests_per_run: int,
    min_interval: float,
) -> Any:
    variant_repository_id = repository_id
    variant_context_database = context_database_path

    def create_config(agent_input: Any) -> CodingSessionConfig:
        inner = OpenAICompatibleProvider(
            replace(
                runtime_provider_config,
                timeout_seconds=PROVIDER_TIMEOUT_SECONDS,
                max_retries=PROVIDER_MAX_RETRIES,
                max_tokens=max_tokens,
            )
        )
        provider = PatientRateLimitedProvider(
            inner,
            max_requests=requests_per_run,
            min_interval_seconds=min_interval,
            backoff_start_seconds=DEFAULT_BACKOFF_START_SECONDS,
            backoff_max_seconds=DEFAULT_BACKOFF_MAX_SECONDS,
            max_attempts=DEFAULT_RATE_LIMIT_ATTEMPTS,
        )
        base = CodingSessionConfig(
            provider=provider,
            model=model,
            storage=JsonlSessionStorage(
                database_path.parent / "tau-sessions" / f"{agent_input.run_id}.jsonl"
            ),
            cwd=cwd,
            provider_name=provider_name,
            owns_initial_provider=True,
            skills_enabled=False,
            extensions_enabled=False,
            project_extensions_enabled=False,
        )
        return build_variant_session_config(
            variant,
            base=base,
            database_path=database_path,
            snapshot_id=snapshot_id,
            repository_id=variant_repository_id,
            context_database_path=variant_context_database,
        )

    return create_config


def _completed_case_ids(db_path: Path) -> set[str]:
    """读取历史实验库中已 COMPLETED 的 case_id（断点续跑）。"""

    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "select case_id from evaluation_agent_runs where status = 'COMPLETED'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _append_progress(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_variant(
    variant: str,
    *,
    cases: list[dict[str, Any]],
    services: SparklePresentationServices,
    repository_id: str,
    snapshot_id: str,
    working_copy_root: Path,
    outdir: Path,
    max_tokens: int,
    requests_per_run: int,
    runs_cap: int,
    min_interval: float,
) -> dict[str, Any]:
    settings = load_provider_settings()
    provider_config = settings.get_provider()
    if not isinstance(provider_config, OpenAICompatibleProviderConfig):
        raise SystemExit("Tau 当前默认 Provider 不是 OpenAI-compatible 配置。")
    model = provider_config.default_model
    credential_store = FileCredentialStore()
    api_key = (
        credential_store.get(provider_config.credential_name)
        if provider_config.credential_name
        else None
    ) or os.environ.get(provider_config.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Tau Provider「{provider_config.name}」没有本地凭据。")
    runtime_provider_config = openai_compatible_config_from_provider(
        provider_config,
        credential_reader=credential_store,
        model=model,
    )
    eval_db = outdir / f"eval-{variant.lower()}.sqlite3"
    completed = _completed_case_ids(eval_db)
    pending = [c for c in cases if c["case_id"] not in completed]
    print(
        f"[{variant}] 共 {len(cases)} 条；已完成跳过 {len(cases) - len(pending)} 条，"
        f"待跑 {len(pending)} 条；模型 {model}",
        flush=True,
    )
    factory = build_live_factory(
        variant=variant,
        runtime_provider_config=runtime_provider_config,
        provider_name=provider_config.name,
        model=model,
        database_path=eval_db,
        snapshot_id=snapshot_id,
        repository_id=repository_id,
        context_database_path=services.database_path,
        cwd=working_copy_root,
        max_tokens=max_tokens,
        requests_per_run=requests_per_run,
        min_interval=min_interval,
    )
    lab = EvaluationLab(
        database_path=eval_db,
        runner=NativeBackend(
            session_config_factory=factory,
            redacted_values=(api_key,),
        ),
        evaluator=Phase3Evaluator(),
    )
    frozen = variant_frozen_summary(
        variant,
        model=model,
        provider_name=provider_config.name,
        max_tokens=max_tokens,
        requests_per_run=requests_per_run,
        timeout_seconds=PROVIDER_TIMEOUT_SECONDS,
        max_retries=PROVIDER_MAX_RETRIES,
        runs_cap=runs_cap,
    )
    frozen["provider_requests_per_run_cap"] = requests_per_run
    frozen["rate_limit_strategy"] = {
        "min_interval_seconds": min_interval,
        "backoff_start_seconds": DEFAULT_BACKOFF_START_SECONDS,
        "backoff_max_seconds": DEFAULT_BACKOFF_MAX_SECONDS,
        "max_attempts": DEFAULT_RATE_LIMIT_ATTEMPTS,
        "resume": "COMPLETED cases skipped on restart",
    }
    progress_path = outdir / f"progress-{variant.lower()}.jsonl"

    for case in pending:
        case_id = case["case_id"]
        print(f"[{variant}] 运行 {case_id} …", flush=True)
        request = EvaluationRequest(
            suite_id=f"phase3-ab-{variant.lower()}",
            snapshot_id=snapshot_id,
            configuration={"frozen": frozen, "repository_id": repository_id},
            cases=(
                EvaluationCase(
                    case_id=case_id,
                    user_input=case["user_input"],
                    ground_truth=case["ground_truth"],
                ),
            ),
        )
        handle = await lab.evaluate(request)
        [event async for event in handle.events()]
        experiment = await handle.snapshot()
        (run,) = experiment.runs
        record = _run_record(run)
        _append_progress(progress_path, record)
        print(
            f"[{variant}] {case_id} → {run.status.value}"
            + (f"（{run.failure_reason}）" if run.failure_reason else ""),
            flush=True,
        )

    return _variant_summary(
        variant=variant,
        cases=cases,
        eval_db=eval_db,
        frozen=frozen,
    )


def _run_record(run: Any) -> dict[str, Any]:
    return {
        "case_id": run.case_id,
        "status": run.status.value,
        "failure_reason": run.failure_reason,
        "failure_message": run.failure_message,
        "final_stop_reason": run.final_stop_reason,
        "usage": _usage_dict(run.usage),
        "timing": _timing_dict(run.timing),
        "output": run.output,
        "results": [
            {"metric": r.metric_name, "value": r.value, "details": dict(r.details)}
            for r in run.results
        ],
    }


def _variant_summary(
    *,
    variant: str,
    cases: list[dict[str, Any]],
    eval_db: Path,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """从实验库读取该变体全部运行，生成逐题记录与汇总。"""

    from sparkle_navigator.persistence.evaluation_store import EvaluationStore

    store = EvaluationStore(eval_db)
    runs_report: list[dict[str, Any]] = []
    conn = sqlite3.connect(eval_db)
    try:
        experiment_ids = [
            row[0]
            for row in conn.execute(
                "select experiment_id from evaluation_experiments order by created_at"
            )
        ]
    finally:
        conn.close()
    for experiment_id in experiment_ids:
        snapshot = store.get_snapshot(experiment_id)
        for run in snapshot.runs:
            runs_report.append(_run_record(run))
    totals: dict[str, Any] = {"runs": len(runs_report)}
    totals["completed"] = sum(
        1 for r in runs_report if r["status"] == AgentRunStatus.COMPLETED.value
    )
    totals["failed"] = sum(1 for r in runs_report if r["failure_reason"])
    scored_metrics = (
        "answer_points_coverage",
        "evidence_precision",
        "evidence_recall",
        "answer_honesty",
    )
    for metric in scored_metrics:
        values = [
            res["value"]
            for r in runs_report
            for res in r["results"]
            if res["metric"] == metric and res["value"] is not None
        ]
        totals[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "n": len(values),
        }
    totals["input_tokens"] = sum(
        (r["usage"] or {}).get("input_tokens", 0) or 0 for r in runs_report
    )
    totals["output_tokens"] = sum(
        (r["usage"] or {}).get("output_tokens", 0) or 0 for r in runs_report
    )
    totals["cost"] = "unknown (provider 未报告计价)"
    return {
        "variant": variant,
        "database": str(eval_db),
        "frozen_config": frozen,
        "totals": totals,
        "runs": runs_report,
        "case_count_expected": len(cases),
    }


def _usage_dict(usage: object) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cost_total": usage.cost_total,
    }


def _timing_dict(timing: object) -> dict[str, Any] | None:
    if timing is None:
        return None
    return {
        "run_duration_ms": timing.run_duration_ms,
        "provider_duration_ms": timing.provider_duration_ms,
        "time_to_first_output_ms": timing.time_to_first_output_ms,
    }


def main() -> None:
    args = parse_args()
    variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    outdir = (
        Path(args.outdir)
        if args.outdir
        else REPO_ROOT / ".scratch" / "phase3-ab"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tau-sessions").mkdir(exist_ok=True)

    manifest, cases = load_phase3_dataset(DATASET_DIR)
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            raise SystemExit(f"--cases 中不存在于数据集的 case_id：{sorted(missing)}")
    if args.limit:
        cases = cases[: args.limit]
    dataset_sha = hashlib.sha256(
        (DATASET_DIR / "cases.json").read_bytes()
    ).hexdigest()

    database, _snapshots, _copies, _reports = default_sparkle_paths(TauPaths().home)
    services = SparklePresentationServices(
        database_path=database.expanduser().resolve(),
        snapshots_root=_snapshots,
        working_copies_root=_copies,
        reports_root=_reports,
    )
    repository_id, snapshot_id, working_copy_root = resolve_tau_repository(services)
    print(f"锁定快照 {snapshot_id[:8]}（commit {LOCKED_SOURCE_COMMIT[:8]}）")

    anchor_reports = verify_evidence_anchors(
        cases,
        database_path=services.database_path,
        repository_id=repository_id,
        snapshot_id=snapshot_id,
    )
    anchor_failures = [r for r in anchor_reports if not r.ok]
    if anchor_failures:
        for report in anchor_failures:
            print(f"锚点核验失败 {report.case_id}: {report.problems}")
        raise SystemExit("证据锚点核验未通过，拒绝运行实验。")
    print(f"锚点核验通过：{len(anchor_reports)} 条 Case")

    variant_reports = []
    for variant in variants:
        report = asyncio.run(
            run_variant(
                variant,
                cases=cases,
                services=services,
                repository_id=repository_id,
                snapshot_id=snapshot_id,
                working_copy_root=working_copy_root,
                outdir=outdir,
                max_tokens=args.max_tokens,
                requests_per_run=args.requests_per_run,
                runs_cap=args.runs_cap,
                min_interval=args.min_interval,
            )
        )
        variant_reports.append(report)
        output_path = outdir / f"report-{variant.lower()}.json"
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"[{variant}] 报告已写入 {output_path}", flush=True)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "dir": str(DATASET_DIR),
            "version": manifest["version"],
            "case_count": len(cases),
            "cases_sha256": dataset_sha,
        },
        "snapshot_id": snapshot_id,
        "source_commit": LOCKED_SOURCE_COMMIT,
        "repository_id": repository_id,
        "variants": [r["totals"] for r in variant_reports],
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"汇总已写入 {summary_path}")


if __name__ == "__main__":
    main()
