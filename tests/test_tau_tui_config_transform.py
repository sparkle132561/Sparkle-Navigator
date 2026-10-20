"""Tau Textual 入口把宿主 CodingSessionConfig 变换传到 Session 准备接缝。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import tau_coding.cli as tau_cli
from tau_coding.session import CodingSessionConfig


def test_tau_textual_entry_forwards_host_session_config_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asyncio.run(
        _tau_textual_entry_forwards_host_session_config_transform(tmp_path, monkeypatch)
    )


async def _tau_textual_entry_forwards_host_session_config_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_tui_app(**kwargs: object) -> str:
        captured.update(kwargs)
        return "tau-session-1"

    monkeypatch.setattr(tau_cli, "run_tui_app", fake_run_tui_app)

    def transform(config: CodingSessionConfig) -> CodingSessionConfig:
        return config

    session_id = await tau_cli.run_openai_tui(
        model=None,
        cwd=tmp_path,
        session_config_transform=transform,
    )

    assert session_id == "tau-session-1"
    assert captured["session_config_transform"] is transform
