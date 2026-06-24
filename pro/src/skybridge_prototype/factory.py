from __future__ import annotations

from typing import Any

from .adapters.anthropic import AnthropicMessagesProvider
from .adapters.http_json import HttpJsonCaptioner, HttpJsonQA, HttpJsonShotDetector, HttpJsonSplitSampler
from .adapters.mock import MockCaptioner, MockQA, MockShotDetector, MockSplitSampler
from .models import ProviderRef
from .providers import CaptionProvider, QAProvider, ShotDetectionProvider, SplitSampleProvider


ROLE_ALIASES = {
    "shot_detection": "shot_detection",
    "split_sample": "split_sample",
    "caption": "caption",
    "qa": "qa",
}


def build_provider(role: str, provider_name: str, config: dict[str, Any]):
    role = ROLE_ALIASES[role]
    provider_config = config["providers"][role][provider_name]
    kind = provider_config.get("type", "mock")
    ref = ProviderRef(
        role=role,
        provider=provider_config.get("provider", provider_name),
        region=provider_config.get("region", "local"),
        endpoint=provider_config.get("endpoint"),
    )

    if kind == "mock":
        return _build_mock(role, ref, provider_config)
    if kind == "http_json":
        return _build_http_json(role, ref, provider_config)
    if kind == "anthropic":
        return AnthropicMessagesProvider(
            ref=ref,
            model=provider_config.get("model"),
            model_env=provider_config.get("model_env", "ANTHROPIC_MODEL"),
            api_key_env=provider_config.get("api_key_env", "ANTHROPIC_API_KEY"),
            timeout_seconds=int(provider_config.get("timeout_seconds", 120)),
        )
    raise ValueError(f"Unsupported provider type for {role}/{provider_name}: {kind}")


def build_workflow_providers(
    config: dict[str, Any],
    selected_overrides: dict[str, str] | None = None,
) -> tuple[ShotDetectionProvider, SplitSampleProvider, CaptionProvider, QAProvider, dict[str, str]]:
    selected = dict(config.get("selected", {}))
    selected.update({key: value for key, value in (selected_overrides or {}).items() if value})
    required = ["shot_detection", "split_sample", "caption", "qa"]
    missing = [role for role in required if role not in selected]
    if missing:
        raise ValueError(f"Missing selected providers for roles: {', '.join(missing)}")
    return (
        build_provider("shot_detection", selected["shot_detection"], config),
        build_provider("split_sample", selected["split_sample"], config),
        build_provider("caption", selected["caption"], config),
        build_provider("qa", selected["qa"], config),
        selected,
    )


def _build_mock(role: str, ref: ProviderRef, config: dict[str, Any]):
    if role == "shot_detection":
        return MockShotDetector(ref, shots_per_video=int(config.get("shots_per_video", 3)))
    if role == "split_sample":
        return MockSplitSampler(ref, samples_per_shot=int(config.get("samples_per_shot", 3)))
    if role == "caption":
        return MockCaptioner(ref)
    if role == "qa":
        return MockQA(ref)
    raise ValueError(f"Unsupported mock role: {role}")


def _build_http_json(role: str, ref: ProviderRef, config: dict[str, Any]):
    timeout = int(config.get("timeout_seconds", 120))
    if role == "shot_detection":
        return HttpJsonShotDetector(ref, timeout_seconds=timeout)
    if role == "split_sample":
        return HttpJsonSplitSampler(
            ref,
            timeout_seconds=timeout,
            samples_per_shot=int(config.get("samples_per_shot", 3)),
        )
    if role == "caption":
        return HttpJsonCaptioner(ref, timeout_seconds=timeout)
    if role == "qa":
        return HttpJsonQA(ref, timeout_seconds=timeout)
    raise ValueError(f"Unsupported HTTP JSON role: {role}")
