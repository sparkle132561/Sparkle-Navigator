"""加载版本化 Benchmark Dataset，并转换为按 Snapshot 隔离的评测请求。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from sparkle_navigator.domain.evaluation import EvaluationCase, EvaluationRequest


@dataclass(frozen=True, slots=True)
class BenchmarkRepository:
    """数据集绑定的公开仓库来源。"""

    repository_id: str
    url: str
    source_commit: str
    license: str


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """公开题目、隐藏 Ground Truth 与生成/核验来源。"""

    case: EvaluationCase
    repository_id: str
    suite_type: str
    question_type: str
    difficulty: str
    provenance: Mapping[str, object]

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def user_input(self) -> str:
        return self.case.user_input

    @property
    def ground_truth(self) -> Mapping[str, object]:
        return self.case.ground_truth


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """已经通过结构和来源校验的只读 Benchmark Dataset。"""

    dataset_id: str
    version: str
    split: str
    repositories: tuple[BenchmarkRepository, ...]
    cases: tuple[BenchmarkCase, ...]

    @classmethod
    def load(cls, manifest_path: Path) -> BenchmarkDataset:
        """加载 Manifest 引用的 Case 文件，并验证身份、来源和隐藏答案结构。"""

        resolved_manifest = Path(manifest_path).resolve()
        manifest = _read_json_object(resolved_manifest, "manifest")
        dataset_id = _required_string(manifest, "dataset_id", "manifest")
        version = _required_string(manifest, "version", "manifest")
        split = _required_string(manifest, "split", "manifest")

        repositories_value = manifest.get("repositories")
        if not isinstance(repositories_value, list) or not repositories_value:
            raise ValueError("manifest.repositories 必须是非空数组。")
        repositories = tuple(
            _parse_repository(value, index)
            for index, value in enumerate(repositories_value)
        )
        repository_ids = tuple(repository.repository_id for repository in repositories)
        if len(repository_ids) != len(set(repository_ids)):
            raise ValueError("manifest.repositories 中 repository_id 不能重复。")

        cases_file = _required_string(manifest, "cases_file", "manifest")
        cases_path = (resolved_manifest.parent / cases_file).resolve()
        if Path(cases_file).is_absolute() or not cases_path.is_relative_to(
            resolved_manifest.parent
        ):
            raise ValueError("manifest.cases_file 必须位于 Manifest 所在目录内。")
        raw_cases = _read_json(cases_path, "cases_file")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("cases_file 必须是非空 Case 数组。")

        case_count = manifest.get("case_count")
        if (
            not isinstance(case_count, int)
            or isinstance(case_count, bool)
            or case_count != len(raw_cases)
        ):
            raise ValueError("manifest.case_count 必须与实际 Case 数量一致。")

        known_repository_ids = set(repository_ids)
        cases = tuple(
            _parse_case(value, index, split, known_repository_ids)
            for index, value in enumerate(raw_cases)
        )
        case_ids = tuple(case.case_id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("cases_file 中 case_id 不能重复。")

        return cls(
            dataset_id=dataset_id,
            version=version,
            split=split,
            repositories=repositories,
            cases=cases,
        )

    def to_evaluation_requests(
        self,
        snapshot_ids: Mapping[str, str],
        repetitions: int,
        configuration: Mapping[str, object],
    ) -> tuple[EvaluationRequest, ...]:
        """按 Repository Snapshot 与 Suite 分组，保留 Ground Truth 在评测侧。"""

        expected_ids = {repository.repository_id for repository in self.repositories}
        supplied_ids = set(snapshot_ids)
        if supplied_ids != expected_ids:
            raise ValueError(
                "snapshot_ids 必须且只能包含 Manifest 中的 repository_id。"
            )
        normalized_snapshot_ids: dict[str, str] = {}
        for repository_id, snapshot_id in snapshot_ids.items():
            if not isinstance(snapshot_id, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", snapshot_id
            ):
                raise ValueError(
                    f"snapshot_ids[{repository_id!r}] 必须是 64 位 SHA-256。"
                )
            normalized_snapshot_ids[repository_id] = snapshot_id.lower()

        grouped: dict[tuple[str, str], list[EvaluationCase]] = {}
        for case in self.cases:
            grouped.setdefault((case.repository_id, case.suite_type), []).append(
                case.case
            )

        return tuple(
            EvaluationRequest(
                suite_id=(
                    f"{self.dataset_id}@{self.version}:{self.split}:"
                    f"{repository_id}:{suite_type}"
                ),
                snapshot_id=normalized_snapshot_ids[repository_id],
                cases=tuple(cases),
                repetitions=repetitions,
                configuration=configuration,
            )
            for (repository_id, suite_type), cases in grouped.items()
        )


def _read_json(path: Path, field: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取有效的 {field}: {path.name}。") from exc


def _read_json_object(path: Path, field: str) -> dict[str, object]:
    value = _read_json(path, field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON object。")
    return value


def _parse_repository(value: object, index: int) -> BenchmarkRepository:
    field = f"manifest.repositories[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON object。")
    repository_id = _required_string(value, "repository_id", field)
    url = _required_string(value, "url", field)
    source_commit = _required_string(value, "source_commit", field)
    license_name = _required_string(value, "license", field)
    if not url.startswith("https://") and not re.fullmatch(
        r"local://[a-z0-9][a-z0-9._-]*",
        url,
    ):
        raise ValueError(
            f"{field}.url 必须是 HTTPS 地址或 local://<repository-id> 标识。"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise ValueError(f"{field}.source_commit 必须是完整 40 位 Commit SHA。")
    return BenchmarkRepository(
        repository_id=repository_id,
        url=url,
        source_commit=source_commit.lower(),
        license=license_name,
    )


def _parse_case(
    value: object,
    index: int,
    split: str,
    repository_ids: set[str],
) -> BenchmarkCase:
    field = f"cases_file[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON object。")
    case_id = _required_string(value, "case_id", field)
    repository_id = _required_string(value, "repository_id", field)
    if repository_id not in repository_ids:
        raise ValueError(f"{field}.repository_id 不存在于 Manifest。")
    suite_type = _required_string(value, "suite_type", field)
    question_type = _required_string(value, "question_type", field)
    difficulty = _required_string(value, "difficulty", field)
    user_input = _required_string(value, "user_input", field)

    ground_truth = value.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError(f"{field}.ground_truth 必须是 JSON object。")
    answer_points = ground_truth.get("answer_points")
    expected_evidence = ground_truth.get("expected_evidence")
    rubric = ground_truth.get("rubric")
    if not isinstance(answer_points, list) or not answer_points or any(
        not isinstance(point, str) or not point.strip() for point in answer_points
    ):
        raise ValueError(f"{field}.ground_truth.answer_points 必须包含非空文本。")
    if not isinstance(expected_evidence, list) or not expected_evidence:
        raise ValueError(f"{field}.ground_truth.expected_evidence 不能为空。")
    for evidence_index, evidence in enumerate(expected_evidence):
        evidence_field = f"{field}.ground_truth.expected_evidence[{evidence_index}]"
        if not isinstance(evidence, dict):
            raise ValueError(f"{evidence_field} 必须是 JSON object。")
        path = evidence.get("path")
        line_start = evidence.get("line_start")
        line_end = evidence.get("line_end")
        if (
            not isinstance(path, str)
            or not path.strip()
            or not isinstance(line_start, int)
            or isinstance(line_start, bool)
            or line_start < 1
            or not isinstance(line_end, int)
            or isinstance(line_end, bool)
            or line_end < line_start
        ):
            raise ValueError(f"{evidence_field} 需要路径和有效的起止行号。")
    if not isinstance(rubric, dict) or not rubric:
        raise ValueError(f"{field}.ground_truth.rubric 必须是非空 JSON object。")

    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{field}.provenance 必须是 JSON object。")
    verification_status = _required_string(
        provenance,
        "verification_status",
        f"{field}.provenance",
    )
    _required_string(provenance, "generator", f"{field}.provenance")
    _required_string(provenance, "generator_version", f"{field}.provenance")
    if split == "gold-candidate" and verification_status != "needs_human_review":
        raise ValueError(
            f"{field} 属于 Gold 候选，verification_status 必须是 needs_human_review。"
        )
    if split == "gold" and verification_status != "human_verified":
        raise ValueError(
            f"{field} 属于 Gold，verification_status 必须是 human_verified。"
        )
    if split == "silver" and verification_status != "auto_verified":
        raise ValueError(
            f"{field} 属于 Silver，verification_status 必须是 auto_verified。"
        )

    return BenchmarkCase(
        case=EvaluationCase(
            case_id=case_id,
            user_input=user_input,
            ground_truth=ground_truth,
        ),
        repository_id=repository_id,
        suite_type=suite_type,
        question_type=question_type,
        difficulty=difficulty,
        provenance=MappingProxyType(dict(provenance)),
    )


def _required_string(value: Mapping[str, object], key: str, field: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{field}.{key} 必须是非空字符串。")
    return item


__all__ = ["BenchmarkCase", "BenchmarkDataset", "BenchmarkRepository"]
