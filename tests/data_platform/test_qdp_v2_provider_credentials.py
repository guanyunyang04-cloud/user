from __future__ import annotations

import json
from datetime import datetime

import pytest

from quantlab.data.qdp_v2 import provider_credentials as credentials


def test_private_profile_round_trip_and_redacted_status(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(credentials.PREFER_ENV_NAME, raising=False)
    path = credentials.save_tushare_provider_profile(
        workspace_root=tmp_path,
        token="unit-secret",
        api_url="https://provider.example/api/",
        plan="week",
        expires_at="2026-08-06T17:48:00+08:00",
        rate_limit_per_minute=150,
        mcp_url_template="https://provider.example/mcp/token={token}",
    )

    profile = credentials.read_tushare_provider_profile(tmp_path)
    status = credentials.tushare_provider_status(
        tmp_path,
        now=datetime.fromisoformat("2026-08-01T12:00:00+08:00"),
    )

    assert path == (
        tmp_path
        / "data"
        / "qdp"
        / "qdp_private"
        / "provider_profiles.json"
    )
    assert profile is not None
    assert profile.token == "unit-secret"
    assert profile.api_url == "https://provider.example/api"
    assert credentials.resolve_tushare_token(tmp_path) == "unit-secret"
    assert credentials.resolve_tushare_api_url(tmp_path) == (
        "https://provider.example/api"
    )
    assert credentials.resolve_tushare_rate_limit(120, workspace_root=tmp_path) == 120
    assert status["credential_source"] == "qdp_private_profile"
    assert status["expired"] is False
    assert "unit-secret" not in json.dumps(status)


def test_environment_overrides_private_profile(tmp_path, monkeypatch) -> None:
    credentials.save_tushare_provider_profile(
        workspace_root=tmp_path,
        token="profile-secret",
        api_url="https://profile.example/api",
        plan="unit",
        expires_at="",
        rate_limit_per_minute=60,
    )
    monkeypatch.setenv("QDP_TUSHARE_PROXY_TOKEN", "environment-secret")
    monkeypatch.setenv("QDP_TUSHARE_API_URL", "https://override.example/api")
    monkeypatch.setenv("QDP_TUSHARE_PREFER_ENV", "1")

    assert credentials.resolve_tushare_token(tmp_path) == "environment-secret"
    assert credentials.resolve_tushare_api_url(tmp_path) == (
        "https://override.example/api"
    )
    assert credentials.resolve_tushare_rate_limit(120, workspace_root=tmp_path) == 60
    assert credentials.tushare_provider_status(tmp_path)["credential_source"] == (
        "environment:QDP_TUSHARE_PROXY_TOKEN"
    )


def test_profile_rejects_non_https_endpoint(tmp_path) -> None:
    with pytest.raises(credentials.ProviderCredentialError, match="must_be_https"):
        credentials.save_tushare_provider_profile(
            workspace_root=tmp_path,
            token="unit-secret",
            api_url="http://provider.example/api",
            plan="unit",
            expires_at="",
            rate_limit_per_minute=60,
        )


def test_save_preserves_other_private_provider_profiles(tmp_path) -> None:
    path = credentials.provider_profiles_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": credentials.PROFILE_SCHEMA,
                "providers": {"another_provider": {"opaque": "value"}},
            }
        ),
        encoding="utf-8",
    )

    credentials.save_tushare_provider_profile(
        workspace_root=tmp_path,
        token="unit-secret",
        api_url="https://provider.example/api",
        plan="unit",
        expires_at="",
        rate_limit_per_minute=60,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["providers"]["another_provider"] == {"opaque": "value"}
    assert credentials.TUSHARE_PROVIDER_ID in payload["providers"]
