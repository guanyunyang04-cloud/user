from __future__ import annotations

"""Local provider profiles for QDP clients.

Provider credentials live below ``quant_data_platform/data/qdp_private`` and
are therefore excluded from Git. The active local profile takes precedence
over legacy environment variables; a command can explicitly request a
process-local environment override.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.manifest import atomic_write_json

PROFILE_SCHEMA = "qdp_private_provider_profiles/v1"
TUSHARE_PROVIDER_ID = "tushare_compatible"
PRIVATE_PROFILE_RELATIVE_PATH = Path("qdp_private/provider_profiles.json")
TOKEN_ENV_NAMES = (
    "QDP_TUSHARE_PROXY_TOKEN",
    "QDP_TUSHARE_TOKEN",
    "TUSHARE_TOKEN",
)
API_URL_ENV_NAMES = ("QDP_TUSHARE_API_URL", "TUSHARE_API_URL")
PREFER_ENV_NAME = "QDP_TUSHARE_PREFER_ENV"


class ProviderCredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class TushareProviderProfile:
    token: str
    api_url: str
    plan: str
    expires_at: str
    rate_limit_per_minute: int
    sdk: str
    mcp_url_template: str
    accept_encoding: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TushareProviderProfile:
        token = str(payload.get("token", "") or "").strip()
        api_url = _validated_https_url(payload.get("api_url", ""), field="api_url")
        rate_limit = int(payload.get("rate_limit_per_minute", 0) or 0)
        if not token:
            raise ProviderCredentialError("tushare_profile_token_missing")
        if rate_limit <= 0:
            raise ProviderCredentialError("tushare_profile_rate_limit_invalid")
        expires_at = str(payload.get("expires_at", "") or "").strip()
        if expires_at:
            try:
                datetime.fromisoformat(expires_at)
            except ValueError as exc:
                raise ProviderCredentialError("tushare_profile_expiry_invalid") from exc
        return cls(
            token=token,
            api_url=api_url,
            plan=str(payload.get("plan", "") or "").strip(),
            expires_at=expires_at,
            rate_limit_per_minute=rate_limit,
            sdk=str(payload.get("sdk", "tushare") or "tushare").strip(),
            mcp_url_template=str(payload.get("mcp_url_template", "") or "").strip(),
            accept_encoding=str(
                payload.get("accept_encoding")
                or dict(payload.get("http_headers", {}) or {}).get("Accept-Encoding")
                or "gzip"
            ).strip(),
        )


def provider_profiles_path(workspace_root: str | Path | None = None) -> Path:
    override = str(os.environ.get("QDP_PROVIDER_PROFILES_PATH", "") or "").strip()
    if override:
        return Path(override).resolve()
    return (
        qdp_paths(workspace_root).data_dir / PRIVATE_PROFILE_RELATIVE_PATH
    ).resolve()


def read_tushare_provider_profile(
    workspace_root: str | Path | None = None,
) -> TushareProviderProfile | None:
    path = provider_profiles_path(workspace_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderCredentialError("provider_profiles_unreadable") from exc
    if str(payload.get("schema", "")) != PROFILE_SCHEMA:
        raise ProviderCredentialError("provider_profiles_schema_changed")
    providers = dict(payload.get("providers", {}) or {})
    record = dict(providers.get(TUSHARE_PROVIDER_ID, {}) or {})
    if not record:
        return None
    return TushareProviderProfile.from_mapping(record)


def save_tushare_provider_profile(
    *,
    token: str,
    api_url: str,
    plan: str,
    expires_at: str,
    rate_limit_per_minute: int,
    workspace_root: str | Path | None = None,
    sdk: str = "tushare",
    mcp_url_template: str = "",
    accept_encoding: str = "gzip",
) -> Path:
    profile = TushareProviderProfile.from_mapping(
        {
            "token": token,
            "api_url": api_url,
            "plan": plan,
            "expires_at": expires_at,
            "rate_limit_per_minute": rate_limit_per_minute,
            "sdk": sdk,
            "mcp_url_template": mcp_url_template,
            "accept_encoding": accept_encoding,
        }
    )
    path = provider_profiles_path(workspace_root)
    providers: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderCredentialError("provider_profiles_unreadable") from exc
        if str(existing.get("schema", "")) != PROFILE_SCHEMA:
            raise ProviderCredentialError("provider_profiles_schema_changed")
        providers = dict(existing.get("providers", {}) or {})
    providers[TUSHARE_PROVIDER_ID] = {
        "token": profile.token,
        "api_url": profile.api_url,
        "plan": profile.plan,
        "expires_at": profile.expires_at,
        "rate_limit_per_minute": profile.rate_limit_per_minute,
        "sdk": profile.sdk,
        "sdk_http_url_override": profile.api_url,
        "http_headers": {"Accept-Encoding": profile.accept_encoding},
        "mcp_url_template": profile.mcp_url_template,
    }
    payload = {
        "schema": PROFILE_SCHEMA,
        "providers": providers,
    }
    atomic_write_json(path, payload)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def resolve_tushare_token(workspace_root: str | Path | None = None) -> str:
    environment = _first_environment_value(TOKEN_ENV_NAMES)
    profile = read_tushare_provider_profile(workspace_root)
    if _prefer_environment() and environment[1]:
        return environment[1]
    if profile is not None:
        return profile.token
    return environment[1]


def resolve_tushare_api_url(workspace_root: str | Path | None = None) -> str:
    environment = _first_environment_value(API_URL_ENV_NAMES)
    profile = read_tushare_provider_profile(workspace_root)
    if _prefer_environment() and environment[1]:
        return _validated_https_url(environment[1], field=environment[0].lower())
    if profile is not None:
        return profile.api_url
    if environment[1]:
        return _validated_https_url(environment[1], field=environment[0].lower())
    raise ProviderCredentialError("tushare_api_url_missing")


def resolve_tushare_rate_limit(
    default: int, *, workspace_root: str | Path | None = None
) -> int:
    requested = max(1, int(default))
    profile = read_tushare_provider_profile(workspace_root)
    if profile is None:
        return requested
    return min(requested, profile.rate_limit_per_minute)


def tushare_credential_values(
    workspace_root: str | Path | None = None,
) -> tuple[str, ...]:
    values = [str(os.environ.get(name, "") or "").strip() for name in TOKEN_ENV_NAMES]
    profile = read_tushare_provider_profile(workspace_root)
    if profile is not None:
        values.append(profile.token)
    return tuple(dict.fromkeys(value for value in values if value))


def tushare_provider_status(
    workspace_root: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    profile = read_tushare_provider_profile(workspace_root)
    environment_token, environment_value = _first_environment_value(TOKEN_ENV_NAMES)
    if profile is None:
        return {
            "provider_id": TUSHARE_PROVIDER_ID,
            "configured": bool(environment_value),
            "credential_source": f"environment:{environment_token}"
            if environment_value
            else "missing",
            "token_present": bool(environment_value),
            "profile_path": str(provider_profiles_path(workspace_root)),
        }
    expired = False
    if profile.expires_at:
        expires = datetime.fromisoformat(profile.expires_at)
        current = now or datetime.now(expires.tzinfo)
        if expires.tzinfo is None and current.tzinfo is not None:
            current = current.replace(tzinfo=None)
        expired = current >= expires
    return {
        "provider_id": TUSHARE_PROVIDER_ID,
        "configured": True,
        "credential_source": f"environment:{environment_token}"
        if _prefer_environment() and environment_value
        else "qdp_private_profile",
        "token_present": True,
        "api_url": profile.api_url,
        "plan": profile.plan,
        "expires_at": profile.expires_at,
        "expired": expired,
        "rate_limit_per_minute": profile.rate_limit_per_minute,
        "sdk": profile.sdk,
        "accept_encoding": profile.accept_encoding,
        "profile_path": str(provider_profiles_path(workspace_root)),
    }


def _validated_https_url(value: Any, *, field: str) -> str:
    text = str(value or "").strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ProviderCredentialError(f"{field}_must_be_https")
    return text


def _prefer_environment() -> bool:
    return str(os.environ.get(PREFER_ENV_NAME, "") or "").strip() == "1"


def _first_environment_value(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return name, value
    return "", ""
