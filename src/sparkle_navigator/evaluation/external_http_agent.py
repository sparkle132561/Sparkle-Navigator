"""通过版本化 JSON HTTP 契约调用外部被测智能体。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath

import httpx

from sparkle_navigator.domain.evaluation import AgentRunInput, AgentRunOutput
from sparkle_navigator.evaluation.core import AgentRunExecutor
from sparkle_navigator.security.redaction import sanitize_trace_text


class ExternalHttpAgentAdapter(AgentRunExecutor):
    """把一个公开 AgentRun 输入转换为 HTTP 请求，并校验 JSON 响应。"""

    def __init__(
        self,
        *,
        endpoint: str,
        client: httpx.AsyncClient,
        timeout_seconds: float = 60.0,
        max_response_bytes: int = 1_000_000,
        redacted_values: tuple[str, ...] = (),
    ) -> None:
        url = httpx.URL(endpoint)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("External HTTP Agent endpoint 必须是有效的 HTTP(S) URL。")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须是有限正数。")
        if (
            not isinstance(max_response_bytes, int)
            or isinstance(max_response_bytes, bool)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes 必须是正整数。")
        self._endpoint = str(url)
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._redacted_values = tuple(value for value in redacted_values if value)

    async def run(self, agent_input: AgentRunInput) -> AgentRunOutput:
        """请求外部 Agent；不发送 Ground Truth，也不替外部 Agent 评分。"""

        payload = {
            "protocol_version": 1,
            "run_id": agent_input.run_id,
            "case_id": agent_input.case_id,
            "repetition": agent_input.repetition,
            "snapshot_id": agent_input.snapshot_id,
            "input": agent_input.user_input,
        }
        async with self._client.stream(
            "POST",
            self._endpoint,
            json=payload,
            headers={"Idempotency-Key": agent_input.run_id},
            timeout=self._timeout_seconds,
        ) as response:
            response.raise_for_status()
            body = await _read_bounded_response(
                response,
                max_bytes=self._max_response_bytes,
            )

        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("External HTTP Agent 必须返回有效 JSON。") from exc
        if not isinstance(data, Mapping):
            raise ValueError("External HTTP Agent 响应必须是 JSON 对象。")
        version = data.get("protocol_version")
        if type(version) is not int or version != 1:
            raise ValueError("External HTTP Agent protocol_version 必须为 1。")
        output = data.get("output")
        if not isinstance(output, str):
            raise ValueError("External HTTP Agent output 必须是字符串。")
        evidence_paths = _validated_evidence_paths(data.get("evidence_paths", []))
        return AgentRunOutput(
            output=sanitize_trace_text(
                output,
                secret_values=self._redacted_values,
            ),
            evidence_paths=evidence_paths,
        )


async def _read_bounded_response(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            raise ValueError("External HTTP Agent 响应超过大小限制。")
        body.extend(chunk)
    return bytes(body)


def _validated_evidence_paths(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("evidence_paths 必须是字符串数组。")
    paths: list[str] = []
    for raw_path in value:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("evidence_paths 中的每条路径都必须是非空字符串。")
        normalized = raw_path.replace("\\", "/")
        segments = normalized.split("/")
        if (
            PurePosixPath(normalized).is_absolute()
            or PureWindowsPath(raw_path).drive
            or any(segment in {"", ".", ".."} for segment in segments)
        ):
            raise ValueError("evidence_paths 只能包含仓库内的相对文件路径。")
        paths.append(normalized)
    return tuple(dict.fromkeys(paths))


__all__ = ["ExternalHttpAgentAdapter"]
