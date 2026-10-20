"""验证真实 Pilot Provider 使用集中配置的单响应输出上限。"""

from __future__ import annotations

from pathlib import Path

from sparkle_navigator.domain.evaluation import AgentRunInput
import sparkle_navigator.evaluation.live_provider as live_provider
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderSettings


def test_tau_live_provider_factory_applies_configured_max_tokens(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider_settings = ProviderSettings(
        default_provider="Agnes API",
        providers=(
            OpenAICompatibleProviderConfig(
                name="Agnes API",
                base_url="https://api.example.invalid/v1",
                credential_name="agnes-test",
                models=("agnes-2.5-flash",),
                default_model="agnes-2.5-flash",
            ),
        ),
    )
    monkeypatch.setattr(
        live_provider,
        "load_provider_settings",
        lambda: provider_settings,
    )
    monkeypatch.setattr(
        live_provider.FileCredentialStore,
        "get",
        lambda _store, _name: "test-credential",
    )
    captured_configs = []

    class CapturingProvider:
        def __init__(self, config) -> None:
            captured_configs.append(config)

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(
        live_provider,
        "OpenAICompatibleProvider",
        CapturingProvider,
    )
    create_session_config, provider_name, model, _credential = (
        live_provider.create_tau_live_session_config_factory(
            tmp_path / "navigator.sqlite3"
        )
    )

    session_config = create_session_config(
        AgentRunInput(
            run_id="test-live-run",
            case_id="test-case",
            repetition=1,
            snapshot_id="a" * 64,
            user_input="Explain the code.",
        )
    )

    assert provider_name == "Agnes API"
    assert model == "agnes-2.5-flash"
    assert session_config.provider is not None
    assert captured_configs[0].max_tokens == live_provider.LIVE_PROVIDER_MAX_TOKENS
    assert live_provider.LIVE_PROVIDER_MAX_TOKENS == 2048
