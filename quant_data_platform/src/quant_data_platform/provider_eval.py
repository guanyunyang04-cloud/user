from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    for _path in (_HERE.parents[1], _HERE.parents[3]):
        _text = str(_path)
        if _text not in sys.path:
            sys.path.insert(0, _text)

from daily_research.data_platform.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    NUMERIC_MARKET_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    normalize_domain_frame,
    normalize_market_frame,
)
from daily_research.data_platform.providers import (
    AkshareEastmoneyProvider,
    BaostockProvider,
    EastmoneyEfinanceProvider,
)


DEFAULT_PROVIDERS = ("akshare", "baostock", "efinance", "mootdx", "cninfo", "current_qdp")
DEFAULT_SYMBOLS = ("000001.SZ", "600000.SH", "300750.SZ", "688001.SH", "000300.SH")
DEFAULT_WINDOWS = (
    ("2010-01-04", "2010-01-15"),
    ("2015-06-01", "2015-06-12"),
    ("2020-03-02", "2020-03-13"),
    ("2024-06-03", "2024-06-14"),
    ("2025-12-01", "2025-12-12"),
    ("2026-06-01", "2026-06-10"),
)
DEFAULT_OUTPUT_ROOT = Path("H:/quant_project/quant_data_platform/data/provider_eval")
MARKET_DAILY_COMPARE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass(frozen=True)
class ProviderEvalConfig:
    providers: tuple[str, ...] = DEFAULT_PROVIDERS
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    windows: tuple[tuple[str, str], ...] = DEFAULT_WINDOWS
    run_tag: str = ""
    output_root: Path = DEFAULT_OUTPUT_ROOT
    use_cache: bool = True
    install_missing: bool = True
    python_executable: str = sys.executable
    qdp_root_manifest: Path = Path("H:/quant_project/quant_data_platform/registry/root_manifest.json")
    canonical_manifest: Path = Path("H:/quant_project/daily_research/output/research_data_lake/canonical/canonical_manifest.json")

    def normalized(self) -> "ProviderEvalConfig":
        run_tag = str(self.run_tag or f"provider_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        providers = tuple(_normalize_provider_name(item) for item in self.providers if str(item or "").strip())
        symbols = tuple(dict.fromkeys(_normalize_symbol(item) for item in self.symbols if str(item or "").strip()))
        windows = tuple((_date_text(start), _date_text(end)) for start, end in self.windows)
        return ProviderEvalConfig(
            providers=providers or DEFAULT_PROVIDERS,
            symbols=symbols or DEFAULT_SYMBOLS,
            windows=windows or DEFAULT_WINDOWS,
            run_tag=run_tag,
            output_root=Path(self.output_root),
            use_cache=bool(self.use_cache),
            install_missing=bool(self.install_missing),
            python_executable=str(self.python_executable or sys.executable),
            qdp_root_manifest=Path(self.qdp_root_manifest),
            canonical_manifest=Path(self.canonical_manifest),
        )


@dataclass(frozen=True)
class ProbeResult:
    provider: str
    endpoint: str
    domain: str = ""
    symbol_count: int = 0
    start_date: str = ""
    end_date: str = ""
    adjusted_flag: str = ""
    network_mode: str = "current_env"
    ok: bool = False
    elapsed_sec: float = 0.0
    rows: int = 0
    columns: tuple[str, ...] = ()
    field_finite_rates: dict[str, float] = field(default_factory=dict)
    duplicate_key_count: int = 0
    error_type: str = ""
    error_message: str = ""
    normalized_sample: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = json.dumps(list(self.columns), ensure_ascii=False)
        payload["field_finite_rates"] = json.dumps(self.field_finite_rates, ensure_ascii=False, sort_keys=True)
        payload["normalized_sample"] = json.dumps(_json_ready(self.normalized_sample), ensure_ascii=False)
        payload["metadata"] = json.dumps(_json_ready(self.metadata), ensure_ascii=False, sort_keys=True)
        payload["rows_per_sec"] = float(self.rows / self.elapsed_sec) if self.elapsed_sec > 0 else np.nan
        return payload


ProviderProbeOverride = Callable[[ProviderEvalConfig, str, Path], tuple[list[ProbeResult], list[pd.DataFrame]]]


def run_provider_eval(
    config: ProviderEvalConfig | None = None,
    *,
    probe_overrides: Mapping[str, ProviderProbeOverride] | None = None,
) -> dict[str, Any]:
    resolved = (config or ProviderEvalConfig()).normalized()
    run_dir = resolved.output_root / resolved.run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(probe_overrides or {})

    all_results: list[ProbeResult] = []
    normalized_market_frames: list[pd.DataFrame] = []
    dependency_status: dict[str, dict[str, Any]] = {}
    start_time = time.perf_counter()

    for provider in resolved.providers:
        if provider == "tushare":
            result = ProbeResult(
                provider=provider,
                endpoint="dependency",
                ok=False,
                error_type="skipped",
                error_message="tushare is intentionally skipped in this provider_eval run",
            )
            all_results.append(result)
            dependency_status[provider] = {"import_ok": False, "install_ok": False, "status": "skipped"}
            continue
        dependency_status[provider] = ensure_provider_dependency(
            provider,
            install_missing=resolved.install_missing,
            python_executable=resolved.python_executable,
        )
        if provider in overrides:
            results, frames = overrides[provider](resolved, provider, cache_dir)
        else:
            results, frames = _run_builtin_provider(resolved, provider=provider, cache_dir=cache_dir)
        all_results.extend(results)
        normalized_market_frames.extend(frames)

    endpoint_results = pd.DataFrame([result.to_record() for result in all_results])
    normalized_market_daily = _concat_frames(normalized_market_frames)
    if not normalized_market_daily.empty:
        normalized_market_daily = normalized_market_daily.sort_values(["trade_date", "symbol", "eval_provider", "adjusted_flag"]).reset_index(drop=True)
    field_coverage = build_field_coverage(normalized_market_daily)
    latency_summary = build_latency_summary(endpoint_results)
    cross_source_diff = build_cross_source_ohlcv_diff(normalized_market_daily)

    write_errors = _write_outputs(
        run_dir=run_dir,
        endpoint_results=endpoint_results,
        normalized_market_daily=normalized_market_daily,
        cross_source_diff=cross_source_diff,
        field_coverage=field_coverage,
        latency_summary=latency_summary,
    )
    elapsed_sec = time.perf_counter() - start_time
    report = {
        "schema_version": 1,
        "run_tag": resolved.run_tag,
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "status": "ok" if not endpoint_results.empty else "empty",
        "config": _config_payload(resolved),
        "summary": {
            "provider_count": len(resolved.providers),
            "endpoint_count": int(len(endpoint_results)),
            "ok_endpoint_count": int(endpoint_results["ok"].sum()) if "ok" in endpoint_results.columns else 0,
            "error_endpoint_count": int((~endpoint_results["ok"].astype(bool)).sum()) if "ok" in endpoint_results.columns else 0,
            "normalized_market_daily_rows": int(len(normalized_market_daily)),
            "cross_source_diff_rows": int(len(cross_source_diff)),
            "elapsed_sec": float(elapsed_sec),
        },
        "dependency_status": dependency_status,
        "write_errors": write_errors,
        "outputs": {
            "provider_eval_report": str((run_dir / "provider_eval_report.json").resolve()),
            "endpoint_results_csv": str((run_dir / "endpoint_results.csv").resolve()),
            "endpoint_results_parquet": str((run_dir / "endpoint_results.parquet").resolve()),
            "normalized_market_daily_csv": str((run_dir / "normalized_market_daily.csv").resolve()),
            "cross_source_ohlcv_diff_csv": str((run_dir / "cross_source_ohlcv_diff.csv").resolve()),
            "field_coverage_csv": str((run_dir / "field_coverage.csv").resolve()),
            "latency_summary_csv": str((run_dir / "latency_summary.csv").resolve()),
            "provider_eval_summary_md": str((run_dir / "provider_eval_summary.md").resolve()),
        },
    }
    (run_dir / "provider_eval_report.json").write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "provider_eval_summary.md").write_text(_summary_markdown(report, endpoint_results, latency_summary), encoding="utf-8")
    return report


def ensure_provider_dependency(
    provider: str,
    *,
    install_missing: bool,
    python_executable: str,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    module_name = {
        "akshare": "akshare",
        "baostock": "baostock",
        "efinance": "efinance",
        "mootdx": "mootdx",
        "cninfo": "requests",
        "current_qdp": "pandas",
    }.get(provider, provider)
    status: dict[str, Any] = {"module": module_name, "import_ok": False, "install_ok": False, "status": "missing"}
    try:
        module = importlib.import_module(module_name)
        status.update({"import_ok": True, "status": "ok", "version": str(getattr(module, "__version__", ""))})
        return status
    except Exception as exc:
        status.update({"import_error_type": type(exc).__name__, "import_error": str(exc)})
    if provider != "mootdx" or not install_missing:
        return status
    cmd = [python_executable, "-m", "pip", "install", "-U", "mootdx[all]"]
    try:
        completed = runner(cmd, capture_output=True, text=True, timeout=900)
        status["install_command"] = " ".join(cmd)
        status["install_returncode"] = int(completed.returncode)
        status["install_stdout_tail"] = str(completed.stdout or "")[-2000:]
        status["install_stderr_tail"] = str(completed.stderr or "")[-2000:]
        status["install_ok"] = completed.returncode == 0
    except Exception as exc:
        status.update({"install_ok": False, "install_error_type": type(exc).__name__, "install_error": str(exc)})
        return status
    if not status["install_ok"]:
        status["status"] = "install_failed"
        return status
    try:
        module = importlib.import_module(module_name)
        status.update({"import_ok": True, "status": "ok", "version": str(getattr(module, "__version__", ""))})
    except Exception as exc:
        status.update({"status": "import_failed_after_install", "import_error_type": type(exc).__name__, "import_error": str(exc)})
    return status


def build_field_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["eval_provider", "endpoint", "adjusted_flag", "field", "rows", "finite_rows", "finite_rate"])
    rows: list[dict[str, Any]] = []
    group_columns = [column for column in ("eval_provider", "endpoint", "adjusted_flag") if column in frame.columns]
    for keys, group in frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        for field_name in [column for column in STANDARD_MARKET_COLUMNS if column in group.columns]:
            series = group[field_name]
            if field_name in NUMERIC_MARKET_COLUMNS:
                finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
            else:
                finite = series.astype(str).str.len().gt(0) & series.notna()
            rows.append(
                {
                    **base,
                    "field": field_name,
                    "rows": int(len(group)),
                    "finite_rows": int(finite.sum()),
                    "finite_rate": float(finite.mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_latency_summary(endpoint_results: pd.DataFrame) -> pd.DataFrame:
    columns = ["provider", "endpoint", "call_count", "ok_count", "p50_latency", "p95_latency", "mean_latency", "max_latency"]
    if endpoint_results is None or endpoint_results.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (provider, endpoint), group in endpoint_results.groupby(["provider", "endpoint"], dropna=False):
        elapsed = pd.to_numeric(group["elapsed_sec"], errors="coerce").dropna()
        rows.append(
            {
                "provider": provider,
                "endpoint": endpoint,
                "call_count": int(len(group)),
                "ok_count": int(group["ok"].astype(bool).sum()),
                "p50_latency": float(elapsed.quantile(0.50)) if not elapsed.empty else np.nan,
                "p95_latency": float(elapsed.quantile(0.95)) if not elapsed.empty else np.nan,
                "mean_latency": float(elapsed.mean()) if not elapsed.empty else np.nan,
                "max_latency": float(elapsed.max()) if not elapsed.empty else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_cross_source_ohlcv_diff(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "eval_provider",
        "symbol",
        "trade_date",
        "adjusted_flag",
        "field",
        "provider_value",
        "current_qdp_value",
        "diff",
        "abs_diff",
        "rel_diff",
        "suggested_unit_factor",
        "unit_adjusted_rel_diff",
    ]
    if frame is None or frame.empty or "eval_provider" not in frame.columns:
        return pd.DataFrame(columns=columns)
    base = frame.loc[frame["eval_provider"].eq("current_qdp")].copy()
    others = frame.loc[~frame["eval_provider"].eq("current_qdp")].copy()
    base = base.loc[base.get("adjusted_flag", "none").astype(str).eq("none")] if not base.empty else base
    others = others.loc[others.get("adjusted_flag", "none").astype(str).eq("none")] if not others.empty else others
    if base.empty or others.empty:
        return pd.DataFrame(columns=columns)
    key = ["symbol", "trade_date"]
    base_cols = key + [column for column in MARKET_DAILY_COMPARE_COLUMNS if column in base.columns]
    base = base.loc[:, base_cols].drop_duplicates(subset=key, keep="first")
    merged = others.merge(base, on=key, how="inner", suffixes=("", "_current_qdp"))
    rows: list[dict[str, Any]] = []
    factors: dict[tuple[str, str], float] = {}
    for provider in sorted(merged["eval_provider"].dropna().astype(str).unique()):
        provider_frame = merged.loc[merged["eval_provider"].astype(str).eq(provider)]
        for field_name in MARKET_DAILY_COMPARE_COLUMNS:
            qdp_field = f"{field_name}_current_qdp"
            if field_name not in provider_frame.columns or qdp_field not in provider_frame.columns:
                continue
            factor = choose_unit_factor(provider_frame[field_name], provider_frame[qdp_field]) if field_name in {"volume", "amount"} else 1.0
            factors[(provider, field_name)] = factor
    for _, row in merged.iterrows():
        provider = str(row.get("eval_provider", ""))
        for field_name in MARKET_DAILY_COMPARE_COLUMNS:
            qdp_field = f"{field_name}_current_qdp"
            if field_name not in merged.columns or qdp_field not in merged.columns:
                continue
            provider_value = _to_float(row.get(field_name))
            qdp_value = _to_float(row.get(qdp_field))
            if np.isnan(provider_value) or np.isnan(qdp_value):
                continue
            diff = provider_value - qdp_value
            factor = float(factors.get((provider, field_name), 1.0))
            adjusted_rel = _safe_rel(provider_value * factor, qdp_value)
            rows.append(
                {
                    "eval_provider": provider,
                    "symbol": str(row.get("symbol", "")),
                    "trade_date": str(row.get("trade_date", "")),
                    "adjusted_flag": str(row.get("adjusted_flag", "")),
                    "field": field_name,
                    "provider_value": provider_value,
                    "current_qdp_value": qdp_value,
                    "diff": diff,
                    "abs_diff": abs(diff),
                    "rel_diff": _safe_rel(provider_value, qdp_value),
                    "suggested_unit_factor": factor,
                    "unit_adjusted_rel_diff": adjusted_rel,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def choose_unit_factor(provider_values: Iterable[Any], qdp_values: Iterable[Any]) -> float:
    provider = pd.to_numeric(pd.Series(provider_values), errors="coerce").astype(float)
    qdp = pd.to_numeric(pd.Series(qdp_values), errors="coerce").astype(float)
    mask = provider.replace([np.inf, -np.inf], np.nan).notna() & qdp.replace([np.inf, -np.inf], np.nan).notna() & qdp.ne(0)
    provider = provider.loc[mask]
    qdp = qdp.loc[mask]
    if provider.empty:
        return 1.0
    factors = (1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000000.0)
    best_factor = 1.0
    best_score = float("inf")
    for factor in factors:
        rel = ((provider * factor) - qdp).abs() / qdp.abs().replace(0, np.nan)
        score = float(rel.median(skipna=True)) if not rel.dropna().empty else float("inf")
        if score < best_score:
            best_score = score
            best_factor = float(factor)
    return best_factor


def _run_builtin_provider(config: ProviderEvalConfig, *, provider: str, cache_dir: Path) -> tuple[list[ProbeResult], list[pd.DataFrame]]:
    if provider == "akshare":
        return _probe_domain_provider(config, provider=provider, provider_obj=AkshareEastmoneyProvider(), cache_dir=cache_dir)
    if provider == "baostock":
        return _probe_domain_provider(config, provider=provider, provider_obj=BaostockProvider(_market_daily_max_workers=1), cache_dir=cache_dir)
    if provider == "efinance":
        return _probe_domain_provider(config, provider=provider, provider_obj=EastmoneyEfinanceProvider(), cache_dir=cache_dir)
    if provider == "mootdx":
        return _probe_mootdx(config, cache_dir=cache_dir)
    if provider == "cninfo":
        return _probe_cninfo(config, cache_dir=cache_dir)
    if provider == "current_qdp":
        return _probe_current_qdp(config, cache_dir=cache_dir)
    return ([ProbeResult(provider=provider, endpoint="provider_dispatch", ok=False, error_type="unsupported_provider", error_message=f"unsupported provider: {provider}")], [])


def _probe_domain_provider(
    config: ProviderEvalConfig,
    *,
    provider: str,
    provider_obj: Any,
    cache_dir: Path,
) -> tuple[list[ProbeResult], list[pd.DataFrame]]:
    results: list[ProbeResult] = []
    frames: list[pd.DataFrame] = []
    import_result = _probe_import(provider)
    results.append(import_result)
    domains = _provider_domains(provider)
    for start_date, end_date in config.windows:
        for adjusted_flag in ("none", "front"):
            endpoint = f"market_daily_{adjusted_flag}"
            probe_results, frame = _cached_or_call(
                config,
                cache_dir=cache_dir,
                provider=provider,
                endpoint=endpoint,
                params={"symbols": config.symbols, "start_date": start_date, "end_date": end_date, "adjusted_flag": adjusted_flag},
                call=lambda network_mode: _call_domain_provider(
                    provider=provider,
                    provider_obj=provider_obj,
                    endpoint=endpoint,
                    domain=DataDomain.MARKET_DAILY,
                    symbols=config.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    adjusted_flag=adjusted_flag,
                    network_mode=network_mode,
                ),
            )
            results.extend(probe_results)
            if not frame.empty:
                frames.append(frame)
        for domain in domains:
            endpoint = str(domain)
            probe_results, frame = _cached_or_call(
                config,
                cache_dir=cache_dir,
                provider=provider,
                endpoint=endpoint,
                params={"symbols": config.symbols, "start_date": start_date, "end_date": end_date, "domain": domain},
                call=lambda network_mode, domain=domain, endpoint=endpoint: _call_domain_provider(
                    provider=provider,
                    provider_obj=provider_obj,
                    endpoint=endpoint,
                    domain=domain,
                    symbols=config.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    adjusted_flag="none",
                    network_mode=network_mode,
                ),
            )
            results.extend(probe_results)
            if domain == DataDomain.MARKET_DAILY and not frame.empty:
                frames.append(frame)
    return results, frames


def _call_domain_provider(
    *,
    provider: str,
    provider_obj: Any,
    endpoint: str,
    domain: str,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    adjusted_flag: str,
    network_mode: str,
) -> tuple[ProbeResult, pd.DataFrame]:
    request = DomainFetchRequest(
        domain=domain,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        adjusted_flag=adjusted_flag,
    )
    start = time.perf_counter()
    try:
        with _clean_proxy_env(enabled=network_mode == "clean_proxy_env"):
            provider_result = provider_obj.fetch_domain(request)
        elapsed = time.perf_counter() - start
        data = provider_result.data.copy() if isinstance(provider_result.data, pd.DataFrame) else pd.DataFrame()
        if domain == DataDomain.MARKET_DAILY:
            data = _tag_frame(data, eval_provider=provider, endpoint=endpoint)
        result = _result_from_frame(
            provider=provider,
            endpoint=endpoint,
            domain=domain,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjusted_flag=adjusted_flag,
            network_mode=network_mode,
            elapsed_sec=elapsed,
            frame=data,
            ok=True,
            metadata={"provider_errors": list(provider_result.error_report or []), "coverage_report": dict(provider_result.coverage_report or {})},
        )
        return result, data if domain == DataDomain.MARKET_DAILY else pd.DataFrame()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        result = ProbeResult(
            provider=provider,
            endpoint=endpoint,
            domain=domain,
            symbol_count=len(symbols),
            start_date=start_date,
            end_date=end_date,
            adjusted_flag=adjusted_flag,
            network_mode=network_mode,
            ok=False,
            elapsed_sec=float(elapsed),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return result, pd.DataFrame()


def _probe_mootdx(config: ProviderEvalConfig, *, cache_dir: Path) -> tuple[list[ProbeResult], list[pd.DataFrame]]:
    results = [_probe_import("mootdx")]
    frames: list[pd.DataFrame] = []
    symbol = next((item for item in config.symbols if item.endswith((".SZ", ".SH")) and not item.startswith("000300")), config.symbols[0])
    start_date, end_date = config.windows[-1]
    for endpoint, call_factory in (
        ("quote_connectivity", lambda network_mode: _call_mootdx_quotes(symbol=symbol, network_mode=network_mode)),
        ("bars_daily_small", lambda network_mode: _call_mootdx_bars(symbol=symbol, network_mode=network_mode)),
    ):
        probe_results, frame = _cached_or_call(
            config,
            cache_dir=cache_dir,
            provider="mootdx",
            endpoint=endpoint,
            params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
            call=call_factory,
        )
        results.extend(probe_results)
        if not frame.empty:
            frames.append(frame)
    return results, frames


def _call_mootdx_quotes(*, symbol: str, network_mode: str) -> tuple[ProbeResult, pd.DataFrame]:
    start = time.perf_counter()
    try:
        with _clean_proxy_env(enabled=network_mode == "clean_proxy_env"):
            from mootdx.quotes import Quotes  # type: ignore

            client = Quotes.factory(market="std")
            try:
                payload = client.quotes(symbol=[_mootdx_symbol(symbol)])
            finally:
                with contextlib.suppress(Exception):
                    client.close()
        elapsed = time.perf_counter() - start
        frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        frame = _normalize_mootdx_quote_frame(frame, symbol=symbol)
        frame = _tag_frame(frame, eval_provider="mootdx", endpoint="quote_connectivity")
        return (
            _result_from_frame(
                provider="mootdx",
                endpoint="quote_connectivity",
                domain="realtime_quote",
                symbols=(symbol,),
                start_date="",
                end_date="",
                adjusted_flag="none",
                network_mode=network_mode,
                elapsed_sec=elapsed,
                frame=frame,
                ok=True,
            ),
            frame,
        )
    except Exception as exc:
        return (
            ProbeResult(
                provider="mootdx",
                endpoint="quote_connectivity",
                domain="realtime_quote",
                symbol_count=1,
                network_mode=network_mode,
                ok=False,
                elapsed_sec=float(time.perf_counter() - start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            ),
            pd.DataFrame(),
        )


def _call_mootdx_bars(*, symbol: str, network_mode: str) -> tuple[ProbeResult, pd.DataFrame]:
    start = time.perf_counter()
    try:
        with _clean_proxy_env(enabled=network_mode == "clean_proxy_env"):
            from mootdx.quotes import Quotes  # type: ignore

            client = Quotes.factory(market="std")
            try:
                payload = client.bars(symbol=_mootdx_symbol(symbol), frequency=9, start=0, offset=32)
            finally:
                with contextlib.suppress(Exception):
                    client.close()
        elapsed = time.perf_counter() - start
        raw = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        if raw.empty:
            frame = normalize_market_frame(pd.DataFrame(), source="mootdx_probe", adjusted_flag="none", require_columns=False)
        else:
            frame = raw.reset_index(drop=True).copy()
            frame["symbol"] = symbol
            frame = normalize_market_frame(frame, source="mootdx_probe", adjusted_flag="none", require_columns=False)
        frame = _tag_frame(frame, eval_provider="mootdx", endpoint="bars_daily_small")
        return (
            _result_from_frame(
                provider="mootdx",
                endpoint="bars_daily_small",
                domain=DataDomain.MARKET_DAILY,
                symbols=(symbol,),
                start_date="",
                end_date="",
                adjusted_flag="none",
                network_mode=network_mode,
                elapsed_sec=elapsed,
                frame=frame,
                ok=True,
            ),
            frame,
        )
    except Exception as exc:
        return (
            ProbeResult(
                provider="mootdx",
                endpoint="bars_daily_small",
                domain=DataDomain.MARKET_DAILY,
                symbol_count=1,
                network_mode=network_mode,
                ok=False,
                elapsed_sec=float(time.perf_counter() - start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            ),
            pd.DataFrame(),
        )


def _probe_cninfo(config: ProviderEvalConfig, *, cache_dir: Path) -> tuple[list[ProbeResult], list[pd.DataFrame]]:
    results = [_probe_import("requests", provider="cninfo")]
    start_date, end_date = config.windows[-2] if len(config.windows) >= 2 else config.windows[0]
    probe_results, frame = _cached_or_call(
        config,
        cache_dir=cache_dir,
        provider="cninfo",
        endpoint="announcement_search",
        params={"symbols": config.symbols[:1], "start_date": start_date, "end_date": end_date},
        call=lambda network_mode: _call_cninfo_announcement_search(config.symbols[0], start_date, end_date, network_mode=network_mode),
    )
    results.extend(probe_results)
    return results, [frame] if not frame.empty and DataDomain.MARKET_DAILY in frame.columns else []


def _call_cninfo_announcement_search(symbol: str, start_date: str, end_date: str, *, network_mode: str) -> tuple[ProbeResult, pd.DataFrame]:
    start = time.perf_counter()
    try:
        import requests

        url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
        headers = {
            "User-Agent": "Mozilla/5.0 provider-eval",
            "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
        }
        payload = {
            "pageNum": 1,
            "pageSize": 5,
            "column": "szse" if symbol.endswith(".SZ") else "sse",
            "tabName": "fulltext",
            "stock": f"{symbol[:6]},",
            "searchkey": "",
            "secid": "",
            "plate": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        with _clean_proxy_env(enabled=network_mode == "clean_proxy_env"):
            response = requests.post(url, headers=headers, data=payload, timeout=20)
        elapsed = time.perf_counter() - start
        if response.status_code >= 400:
            raise RuntimeError(f"cninfo_http_{response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"cninfo_non_json_response: {response.text[:200]}") from exc
        announcements = data.get("announcements", []) if isinstance(data, dict) else []
        frame = pd.DataFrame(announcements)
        if not frame.empty:
            frame = frame.rename(columns={"announcementTitle": "title", "announcementTime": "trade_date", "adjunctUrl": "url"})
            if "trade_date" in frame.columns:
                values = pd.to_numeric(frame["trade_date"], errors="coerce")
                frame["trade_date"] = pd.to_datetime(values, unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")
            frame["symbol"] = symbol
            frame["source"] = "cninfo"
            frame = normalize_domain_frame(frame, domain=DataDomain.ANNOUNCEMENT, source="cninfo", as_of_date=end_date, require_columns=False)
        result = _result_from_frame(
            provider="cninfo",
            endpoint="announcement_search",
            domain=DataDomain.ANNOUNCEMENT,
            symbols=(symbol,),
            start_date=start_date,
            end_date=end_date,
            adjusted_flag="",
            network_mode=network_mode,
            elapsed_sec=elapsed,
            frame=frame,
            ok=True,
            metadata={"capability": "announcement_search"},
        )
        return result, pd.DataFrame()
    except Exception as exc:
        message = str(exc)
        error_type = "capability_blocked" if "token" in message.lower() or "sign" in message.lower() else type(exc).__name__
        return (
            ProbeResult(
                provider="cninfo",
                endpoint="announcement_search",
                domain=DataDomain.ANNOUNCEMENT,
                symbol_count=1,
                start_date=start_date,
                end_date=end_date,
                network_mode=network_mode,
                ok=False,
                elapsed_sec=float(time.perf_counter() - start),
                error_type=error_type,
                error_message=message,
            ),
            pd.DataFrame(),
        )


def _probe_current_qdp(config: ProviderEvalConfig, *, cache_dir: Path) -> tuple[list[ProbeResult], list[pd.DataFrame]]:
    results = [_probe_import("pandas", provider="current_qdp")]
    start_date = min(start for start, _ in config.windows)
    end_date = max(end for _, end in config.windows)
    probe_results, frame = _cached_or_call(
        config,
        cache_dir=cache_dir,
        provider="current_qdp",
        endpoint="market_daily_baseline",
        params={"symbols": config.symbols, "start_date": start_date, "end_date": end_date},
        call=lambda network_mode: _call_current_qdp(config, start_date=start_date, end_date=end_date, network_mode=network_mode),
    )
    results.extend(probe_results)
    return results, [frame] if not frame.empty else []


def _call_current_qdp(config: ProviderEvalConfig, *, start_date: str, end_date: str, network_mode: str) -> tuple[ProbeResult, pd.DataFrame]:
    del network_mode
    start = time.perf_counter()
    try:
        root_manifest = _read_json(config.qdp_root_manifest)
        canonical_manifest_path = Path(str(root_manifest.get("canonical_manifest") or config.canonical_manifest))
        canonical_manifest = _read_json(canonical_manifest_path)
        dataset_id = str(canonical_manifest.get("canonical_dataset_id") or root_manifest.get("canonical_dataset_id") or "")
        if not dataset_id.startswith("policy_input_bundle__"):
            raise RuntimeError(f"cannot resolve policy_input_bundle dataset id from {canonical_manifest_path}")
        fingerprint = dataset_id.split("__", 1)[1]
        bundle_manifest_path = (
            canonical_manifest_path.parents[1]
            / "parquet"
            / "bronze_silver"
            / "policy_input_bundle"
            / fingerprint
            / "bundle_manifest.json"
        )
        bundle_manifest = _read_json(bundle_manifest_path)
        path_glob = str(dict(bundle_manifest.get("market_daily_source_paths", {}) or {}).get("silver_domain_data", "") or "")
        if not path_glob:
            raise RuntimeError(f"bundle manifest has no market_daily_source_paths.silver_domain_data: {bundle_manifest_path}")
        paths = sorted(Path().glob(path_glob)) if not any(ch in path_glob for ch in "*?[") else []
        if not paths:
            import glob

            paths = [Path(item) for item in sorted(glob.glob(path_glob))]
        frames: list[pd.DataFrame] = []
        filters = [("symbol", "in", list(config.symbols)), ("trade_date", ">=", start_date), ("trade_date", "<=", end_date)]
        for path in paths:
            try:
                part = pd.read_parquet(path, filters=filters)
            except Exception:
                part = pd.read_parquet(path)
                if not part.empty:
                    part = part.loc[
                        part["symbol"].astype(str).isin(set(config.symbols))
                        & part["trade_date"].astype(str).ge(start_date)
                        & part["trade_date"].astype(str).le(end_date)
                    ]
            if not part.empty:
                frames.append(part)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        frame = normalize_market_frame(raw, source="current_qdp", adjusted_flag="none", require_columns=False)
        frame = _tag_frame(frame, eval_provider="current_qdp", endpoint="market_daily_baseline")
        elapsed = time.perf_counter() - start
        return (
            _result_from_frame(
                provider="current_qdp",
                endpoint="market_daily_baseline",
                domain=DataDomain.MARKET_DAILY,
                symbols=config.symbols,
                start_date=start_date,
                end_date=end_date,
                adjusted_flag="none",
                network_mode="local_disk",
                elapsed_sec=elapsed,
                frame=frame,
                ok=True,
                metadata={"dataset_id": dataset_id, "bundle_manifest": str(bundle_manifest_path)},
            ),
            frame,
        )
    except Exception as exc:
        return (
            ProbeResult(
                provider="current_qdp",
                endpoint="market_daily_baseline",
                domain=DataDomain.MARKET_DAILY,
                symbol_count=len(config.symbols),
                start_date=start_date,
                end_date=end_date,
                adjusted_flag="none",
                network_mode="local_disk",
                ok=False,
                elapsed_sec=float(time.perf_counter() - start),
                error_type=type(exc).__name__,
                error_message=str(exc),
            ),
            pd.DataFrame(),
        )


def _cached_or_call(
    config: ProviderEvalConfig,
    *,
    cache_dir: Path,
    provider: str,
    endpoint: str,
    params: Mapping[str, Any],
    call: Callable[[str], tuple[ProbeResult, pd.DataFrame]],
) -> tuple[list[ProbeResult], pd.DataFrame]:
    key = _cache_key(provider=provider, endpoint=endpoint, params=params)
    cache_path = cache_dir / f"{key}.json"
    if config.use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if "results" in payload:
            results = [_probe_result_from_payload(item) for item in payload.get("results", [])]
        else:
            results = [_probe_result_from_payload(payload["result"])]
        frame = pd.DataFrame(payload.get("normalized_market_records", []))
        return results, frame
    result, frame = call("current_env")
    results = [result]
    if not result.ok and _looks_like_proxy_error(result.error_type, result.error_message):
        retry_result, retry_frame = call("clean_proxy_env")
        results.append(retry_result)
        result, frame = retry_result, retry_frame
    if config.use_cache:
        cache_payload = {
            "results": [_json_ready(asdict(item)) for item in results],
            "normalized_market_records": _frame_records(frame),
        }
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, frame


def _result_from_frame(
    *,
    provider: str,
    endpoint: str,
    domain: str,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    adjusted_flag: str,
    network_mode: str,
    elapsed_sec: float,
    frame: pd.DataFrame,
    ok: bool,
    metadata: Mapping[str, Any] | None = None,
) -> ProbeResult:
    frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    is_ok = bool(ok)
    error_type = ""
    error_message = ""
    if is_ok and frame.empty:
        is_ok = False
        error_type = "empty_data"
        error_message = "provider returned no rows"
    return ProbeResult(
        provider=provider,
        endpoint=endpoint,
        domain=domain,
        symbol_count=len(symbols),
        start_date=start_date,
        end_date=end_date,
        adjusted_flag=adjusted_flag,
        network_mode=network_mode,
        ok=is_ok,
        elapsed_sec=float(elapsed_sec),
        rows=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
        field_finite_rates=_field_finite_rates(frame),
        duplicate_key_count=_duplicate_key_count(frame, domain=domain),
        error_type=error_type,
        error_message=error_message,
        normalized_sample=_frame_records(frame.head(3)),
        metadata=dict(metadata or {}),
    )


def _probe_result_from_payload(payload: Mapping[str, Any]) -> ProbeResult:
    values = dict(payload)
    values["columns"] = tuple(values.get("columns", ()) or ())
    values["field_finite_rates"] = dict(values.get("field_finite_rates", {}) or {})
    values["normalized_sample"] = list(values.get("normalized_sample", []) or [])
    values["metadata"] = dict(values.get("metadata", {}) or {})
    return ProbeResult(**values)


def _probe_import(module_name: str, *, provider: str | None = None) -> ProbeResult:
    provider_name = provider or module_name
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        return ProbeResult(
            provider=provider_name,
            endpoint="dependency",
            ok=True,
            elapsed_sec=float(time.perf_counter() - start),
            rows=1,
            columns=("module", "version"),
            metadata={"module": module_name, "version": str(getattr(module, "__version__", ""))},
        )
    except Exception as exc:
        return ProbeResult(
            provider=provider_name,
            endpoint="dependency",
            ok=False,
            elapsed_sec=float(time.perf_counter() - start),
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"module": module_name},
        )


def _provider_domains(provider: str) -> tuple[str, ...]:
    if provider == "akshare":
        return (DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION, DataDomain.LIMIT_STATUS, DataDomain.MONEY_FLOW_HOTSPOT)
    if provider == "baostock":
        return (DataDomain.VALUATION, DataDomain.INDUSTRY_CONCEPT, DataDomain.INDEX_CONSTITUENTS, DataDomain.TRADING_CALENDAR)
    if provider == "efinance":
        return (DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION)
    return ()


def _normalize_mootdx_quote_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return normalize_market_frame(pd.DataFrame(), source="mootdx_probe", adjusted_flag="none", require_columns=False)
    data = frame.copy()
    data["symbol"] = symbol
    data["trade_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    for target, aliases in {
        "open": ("open", "开盘", "open_price"),
        "high": ("high", "最高", "high_price"),
        "low": ("low", "最低", "low_price"),
        "close": ("price", "close", "最新", "last_close"),
        "volume": ("vol", "volume", "成交量"),
        "amount": ("amount", "成交额"),
    }.items():
        if target in data.columns:
            continue
        for alias in aliases:
            if alias in data.columns:
                data[target] = data[alias]
                break
    return normalize_market_frame(data, source="mootdx_probe", adjusted_flag="none", require_columns=False)


def _tag_frame(frame: pd.DataFrame, *, eval_provider: str, endpoint: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    out["eval_provider"] = eval_provider
    out["endpoint"] = endpoint
    return out


def _field_finite_rates(frame: pd.DataFrame) -> dict[str, float]:
    if frame is None or frame.empty:
        return {}
    rates: dict[str, float] = {}
    for column in frame.columns:
        series = frame[column]
        if column in NUMERIC_MARKET_COLUMNS:
            finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        else:
            finite = series.notna()
        rates[str(column)] = float(finite.mean()) if len(series) else np.nan
    return rates


def _duplicate_key_count(frame: pd.DataFrame, *, domain: str) -> int:
    if frame is None or frame.empty:
        return 0
    if domain == DataDomain.MARKET_DAILY:
        subset = [column for column in ("symbol", "trade_date", "adjusted_flag", "eval_provider") if column in frame.columns]
    elif "trade_date" in frame.columns and "symbol" in frame.columns:
        subset = [column for column in ("symbol", "trade_date", "eval_provider") if column in frame.columns]
    else:
        return 0
    if not subset:
        return 0
    return int(frame.duplicated(subset=subset).sum())


def _write_outputs(
    *,
    run_dir: Path,
    endpoint_results: pd.DataFrame,
    normalized_market_daily: pd.DataFrame,
    cross_source_diff: pd.DataFrame,
    field_coverage: pd.DataFrame,
    latency_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    endpoint_results.to_csv(run_dir / "endpoint_results.csv", index=False, encoding="utf-8-sig")
    normalized_market_daily.to_csv(run_dir / "normalized_market_daily.csv", index=False, encoding="utf-8-sig")
    cross_source_diff.to_csv(run_dir / "cross_source_ohlcv_diff.csv", index=False, encoding="utf-8-sig")
    field_coverage.to_csv(run_dir / "field_coverage.csv", index=False, encoding="utf-8-sig")
    latency_summary.to_csv(run_dir / "latency_summary.csv", index=False, encoding="utf-8-sig")
    try:
        endpoint_results.to_parquet(run_dir / "endpoint_results.parquet", index=False)
    except Exception as exc:
        errors.append({"output": "endpoint_results.parquet", "error_type": type(exc).__name__, "message": str(exc)})
    return errors


def _summary_markdown(report: Mapping[str, Any], endpoint_results: pd.DataFrame, latency_summary: pd.DataFrame) -> str:
    summary = dict(report.get("summary", {}) or {})
    lines = [
        f"# Provider Eval Summary: {report.get('run_tag', '')}",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Providers: {summary.get('provider_count', 0)}",
        f"- Endpoints: {summary.get('endpoint_count', 0)}",
        f"- OK endpoints: {summary.get('ok_endpoint_count', 0)}",
        f"- Error endpoints: {summary.get('error_endpoint_count', 0)}",
        f"- Normalized market daily rows: {summary.get('normalized_market_daily_rows', 0)}",
        f"- Cross-source diff rows: {summary.get('cross_source_diff_rows', 0)}",
        "",
        "## Endpoint Status",
        "",
    ]
    if endpoint_results.empty:
        lines.append("No endpoint rows.")
    else:
        cols = ["provider", "endpoint", "ok", "rows", "elapsed_sec", "error_type", "error_message"]
        lines.append(endpoint_results.loc[:, [c for c in cols if c in endpoint_results.columns]].to_markdown(index=False))
    lines.extend(["", "## Latency", ""])
    if latency_summary.empty:
        lines.append("No latency rows.")
    else:
        lines.append(latency_summary.to_markdown(index=False))
    return "\n".join(lines) + "\n"


def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    cleaned = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not cleaned:
        columns = [*STANDARD_MARKET_COLUMNS, "eval_provider", "endpoint"]
        return pd.DataFrame(columns=columns)
    return pd.concat(cleaned, ignore_index=True)


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return list(_json_ready(frame.replace({np.nan: None}).to_dict(orient="records")))


def _config_payload(config: ProviderEvalConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_root"] = str(config.output_root)
    payload["qdp_root_manifest"] = str(config.qdp_root_manifest)
    payload["canonical_manifest"] = str(config.canonical_manifest)
    return _json_ready(payload)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _cache_key(*, provider: str, endpoint: str, params: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_ready({"provider": provider, "endpoint": endpoint, "params": params}), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _looks_like_proxy_error(error_type: str, message: str) -> bool:
    text = f"{error_type} {message}".lower()
    return "proxy" in text or "maxretryerror" in text or "unable to connect to proxy" in text


@contextlib.contextmanager
def _clean_proxy_env(*, enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    saved = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    try:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_provider_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "ak": "akshare",
        "bs": "baostock",
        "eastmoney": "efinance",
        "eastmoney_efinance": "efinance",
        "qdp": "current_qdp",
        "current": "current_qdp",
    }
    return aliases.get(normalized, normalized)


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if "." in raw:
        code, suffix = raw.split(".", 1)
        return f"{code.zfill(6)}.{suffix}"
    code = raw.zfill(6)
    if code.startswith(("5", "6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _mootdx_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol).split(".", 1)[0]


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_rel(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right) or right == 0:
        return float("nan")
    return float((left - right) / abs(right))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _parse_windows(value: str) -> tuple[tuple[str, str], ...]:
    windows: list[tuple[str, str]] = []
    for item in _split_csv(value):
        if ":" not in item:
            raise ValueError(f"window must be START:END, got {item}")
        start, end = item.split(":", 1)
        windows.append((_date_text(start), _date_text(end)))
    return tuple(windows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only provider evaluation probes.")
    parser.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--windows", default=",".join(f"{start}:{end}" for start, end in DEFAULT_WINDOWS))
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-install-missing", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_provider_eval(
        ProviderEvalConfig(
            providers=_split_csv(args.providers),
            symbols=_split_csv(args.symbols),
            windows=_parse_windows(args.windows),
            run_tag=args.run_tag,
            output_root=Path(args.output_root),
            use_cache=not bool(args.no_cache),
            install_missing=not bool(args.no_install_missing),
            python_executable=str(args.python_executable or sys.executable),
        )
    )
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2 if args.json else None))
    return 0 if report.get("status") in {"ok", "empty"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
