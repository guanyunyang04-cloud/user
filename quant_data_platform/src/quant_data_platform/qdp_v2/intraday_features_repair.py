from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.domains.contracts import DataDomain, build_intraday_daily_feature_frame
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


def repair_intraday_daily_features(
    *,
    workspace_root: str | Path | None = None,
    issue_path: str | Path,
    source_dataset_id: str = "",
    one_minute_dataset_id: str = "",
    five_minute_dataset_id: str = "",
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    activate_domain: bool = False,
    copy_mode: str = "hardlink",
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    source_id = source_dataset_id or str(dict(active.get("derived", {}) or {}).get("intraday_daily_features", "") or "")
    one_id = one_minute_dataset_id or str(dict(active.get("raw", {}) or {}).get("market_intraday_1m", "") or "")
    five_id = five_minute_dataset_id or str(dict(active.get("raw", {}) or {}).get("market_intraday_5m", "") or "")
    source_manifest = _manifest(root, source_id, "intraday_daily_features")
    one_manifest = _manifest(root, one_id, "market_intraday_1m")
    five_manifest = _manifest(root, five_id, "market_intraday_5m")
    source = read_dataset_manifest(source_manifest)
    one = read_dataset_manifest(one_manifest)
    five = read_dataset_manifest(five_manifest)
    issues = pd.read_csv(issue_path)
    if issues.empty:
        return {"status": "ok", "message": "issue_file_empty"}
    issues["symbol"] = issues["symbol"].astype(str).str.upper()
    issues["trade_date"] = issues["trade_date"].astype(str)
    all_keys = issues[["symbol", "trade_date"]].drop_duplicates().reset_index(drop=True)
    replace_keys = issues.loc[issues["repair_class"].eq("traded_day_partial_zero_bars"), ["symbol", "trade_date"]].drop_duplicates().reset_index(drop=True)
    profile = resolve_runtime_profile(runtime)
    memory_limit = duckdb_memory_limit or profile.duckdb_memory_limit
    issue_hash = stable_hash({"source": source.dataset_id, "issue_path": str(issue_path), "one": one.dataset_id, "five": five.dataset_id})
    target_dataset_id = f"intraday_daily_features__{stable_hash({'source': source.dataset_id, 'repair': 'intraday_zero_v1', 'issue_hash': issue_hash})}"
    target_dir = root / "datasets" / "intraday_daily_features" / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    feature_locations = _feature_key_locations(root=root, manifest=source, keys=all_keys, memory_limit=memory_limit, threads=profile.duckdb_threads)
    affected_paths = sorted(set(feature_locations["feature_path"].astype(str))) if not feature_locations.empty else []
    replacements = _build_feature_replacements(
        root=root,
        issues=issues,
        replace_keys=replace_keys,
        one=one,
        five=five,
        source_one_for_index=_manifest(root, str(issues.attrs.get("source_dataset_id", "")), "market_intraday_1m") if False else None,
        memory_limit=memory_limit,
        threads=profile.duckdb_threads,
        source_columns=[item["name"] for item in source.schema],
    )
    replacements = replacements.merge(
        feature_locations[["symbol", "trade_date", "feature_path"]].drop_duplicates(),
        on=["symbol", "trade_date"],
        how="left",
    )
    unresolved_replacements = replacements.loc[replacements["feature_path"].isna(), ["symbol", "trade_date"]].drop_duplicates()
    if not unresolved_replacements.empty:
        return {
            "status": "blocked",
            "reason": "replacement_feature_shard_missing",
            "examples": unresolved_replacements.head(20).to_dict("records"),
        }
    affected_set = set(affected_paths)
    entries: list[ShardManifestEntry] = []
    rewritten = 0
    linked = 0
    removed_rows = 0
    inserted_rows = 0
    source_columns = [item["name"] for item in source.schema]
    for shard in source.shards:
        source_path = _resolve(root, shard.path)
        target_path = staging_shards / Path(shard.path).name
        if str(source_path.resolve()) in affected_set:
            shard_keys = all_keys.merge(
                feature_locations.loc[feature_locations["feature_path"].eq(str(source_path.resolve())), ["symbol", "trade_date"]].drop_duplicates(),
                on=["symbol", "trade_date"],
                how="inner",
            )
            shard_replacements = replacements.loc[replacements["feature_path"].eq(str(source_path.resolve()))].drop(columns=["feature_path"], errors="ignore")
            stats = _rewrite_feature_shard(
                source_path=source_path,
                target_path=target_path,
                remove_keys=shard_keys,
                replacements=shard_replacements,
                columns=source_columns,
            )
            rewritten += 1
            removed_rows += int(stats["removed_rows"])
            inserted_rows += int(stats["inserted_rows"])
            entry = _entry(target_dataset_id=target_dataset_id, target_path=target_path, source_path=source_path, source_shard=shard, row_count=int(stats["row_count"]))
        else:
            _copy_or_link(source_path, target_path, mode=copy_mode)
            linked += 1
            entry = ShardManifestEntry(
                path=f"datasets/intraday_daily_features/{target_dataset_id}/shards/{target_path.name}",
                row_count=shard.row_count,
                start_date=shard.start_date,
                end_date=shard.end_date,
                status="stored",
                file_size=int(target_path.stat().st_size),
                schema_hash=shard.schema_hash,
                source_path=str(source_path.resolve()),
                content_key=shard.content_key,
                metadata={**dict(shard.metadata or {}), "repair_copy_mode": copy_mode},
            )
        entries.append(entry)
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)
    schema = source.schema
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="intraday_daily_features",
        layer="derived",
        frequency="1d",
        contract_version="qdp_v2_intraday_daily_features_v1",
        primary_key=["trade_date", "symbol"],
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={
            "provider": "qdp_v2",
            "created_by": "repair_intraday_daily_features",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "one_minute_dataset_id": one.dataset_id,
            "five_minute_dataset_id": five.dataset_id,
            "issue_path": str(Path(issue_path).resolve()),
        },
        quality={"path_refs_exist": True, "primary_key_unique": "not_checked_after_repair", "intraday_zero_repair": "applied"},
        notes=["incrementally repaired after 1m/5m zero-price cleanup"],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        derived = dict(active.get("derived", {}) or {})
        derived["intraday_daily_features"] = target_dataset_id
        active["derived"] = derived
        active["updated_at"] = utc_now()
        active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "source_dataset_id": source.dataset_id,
        "one_minute_dataset_id": one.dataset_id,
        "five_minute_dataset_id": five.dataset_id,
        "issue_path": str(Path(issue_path).resolve()),
        "affected_feature_shards": len(affected_set),
        "rewritten_shards": rewritten,
        "linked_shards": linked,
        "removed_rows": removed_rows,
        "inserted_rows": inserted_rows,
        "row_count": manifest.row_count,
        "active_manifest": active_path,
    }
    atomic_write_json(root / "runs" / f"repair_intraday_daily_features_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def _build_feature_replacements(
    *,
    root: Path,
    issues: pd.DataFrame,
    replace_keys: pd.DataFrame,
    one: DatasetManifest,
    five: DatasetManifest,
    source_one_for_index: Path | None,
    memory_limit: str,
    threads: int,
    source_columns: list[str],
) -> pd.DataFrame:
    if replace_keys.empty:
        return pd.DataFrame(columns=source_columns)
    issue_paths = issues["shard_path"].dropna().astype(str).unique().tolist()
    five_paths = _paths_for_issue_filenames(root=root, issue_paths=issue_paths, domain_manifest=five, target_domain="market_intraday_5m")
    one_paths = _paths_for_issue_filenames(root=root, issue_paths=issue_paths, domain_manifest=one, target_domain="market_intraday_1m")
    five_rows = _read_joined_intraday(root=root, paths=five_paths, keys=replace_keys, memory_limit=memory_limit, threads=threads)
    if five_rows.empty:
        return pd.DataFrame(columns=source_columns)
    features = build_intraday_daily_feature_frame(five_rows, source="qdp_v2", adjusted_flag="none")
    features = _override_open_close_auction_features(
        features=features,
        one_rows=_read_joined_intraday(root=root, paths=one_paths, keys=replace_keys, memory_limit=memory_limit, threads=threads),
        five_rows=five_rows,
    )
    features = _override_open_gap_from_daily(root=root, features=features, five_rows=five_rows, memory_limit=memory_limit)
    for column in source_columns:
        if column not in features.columns:
            features[column] = pd.NA
    return features.loc[:, source_columns].copy()


def _feature_key_locations(*, root: Path, manifest: DatasetManifest, keys: pd.DataFrame, memory_limit: str, threads: int) -> pd.DataFrame:
    import duckdb  # type: ignore

    paths = [str(_resolve(root, item.path)) for item in manifest.shards]
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads or 1))}")
        con.register("keys", keys)
        return con.execute(
            """
            select distinct filename as feature_path, f.symbol, f.trade_date
            from read_parquet(?, union_by_name=true, filename=true) f
            inner join keys k on f.symbol = k.symbol and f.trade_date = k.trade_date
            """,
            [paths],
        ).fetchdf().assign(feature_path=lambda df: df["feature_path"].map(lambda value: str(Path(str(value)).resolve())))


def _paths_for_issue_filenames(*, root: Path, issue_paths: list[str], domain_manifest: DatasetManifest, target_domain: str) -> list[Path]:
    by_name = {Path(item.path).name: _resolve(root, item.path) for item in domain_manifest.shards}
    output: list[Path] = []
    seen: set[str] = set()
    for path in issue_paths:
        name = Path(path).name
        target_name = name
        if target_domain == "market_intraday_5m":
            target_name = name.replace("market_intraday_1m", "market_intraday_5m")
        candidate = by_name.get(target_name)
        if candidate is not None and str(candidate) not in seen:
            output.append(candidate)
            seen.add(str(candidate))
    if output:
        return output
    return _paths_for_issue_shard_indexes(root=root, issue_paths=issue_paths, domain_manifest=domain_manifest)


def _paths_for_issue_shard_indexes(*, root: Path, issue_paths: list[str], domain_manifest: DatasetManifest) -> list[Path]:
    indexes = set()
    for path in issue_paths:
        name = Path(path).name
        parts = name.split("_")
        for part in parts:
            if part.isdigit():
                indexes.add(int(part))
                break
    paths: list[Path] = []
    for index in sorted(indexes):
        if index < len(domain_manifest.shards):
            paths.append(_resolve(root, domain_manifest.shards[index].path))
    return paths


def _read_joined_intraday(*, root: Path, paths: list[Path], keys: pd.DataFrame, memory_limit: str, threads: int) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads or 1))}")
        con.register("keys", keys)
        return con.execute(
            """
            select d.*
            from read_parquet(?, union_by_name=true) d
            inner join keys k on d.symbol = k.symbol and d.trade_date = k.trade_date
            order by d.trade_date, d.symbol, d.bar_time
            """,
            [[str(path) for path in paths]],
        ).fetchdf()


def _override_open_close_auction_features(*, features: pd.DataFrame, one_rows: pd.DataFrame, five_rows: pd.DataFrame) -> pd.DataFrame:
    if features.empty or one_rows.empty:
        return features
    out = features.copy()
    totals = five_rows.groupby(["symbol", "trade_date"], dropna=False)["amount"].sum().rename("total_amount").reset_index()
    auction = one_rows.loc[one_rows["bar_time"].astype(str).str.zfill(9).isin(["093100000", "150000000"])].copy()
    if auction.empty:
        return out
    auction["bar_time"] = auction["bar_time"].astype(str).str.zfill(9)
    auction = auction.merge(totals, on=["symbol", "trade_date"], how="left")
    for prefix, bar_time in (("opening_auction", "093100000"), ("closing_auction", "150000000")):
        sub = auction.loc[auction["bar_time"].eq(bar_time)].copy()
        if sub.empty:
            continue
        sub[f"{prefix}_ret"] = _safe_return_series(sub["close"], sub["open"])
        sub[f"{prefix}_amount"] = pd.to_numeric(sub["amount"], errors="coerce")
        sub[f"{prefix}_volume"] = pd.to_numeric(sub["volume"], errors="coerce")
        sub[f"{prefix}_amount_share"] = sub[f"{prefix}_amount"] / pd.to_numeric(sub["total_amount"], errors="coerce").where(pd.to_numeric(sub["total_amount"], errors="coerce") > 0)
        sub[f"{prefix}_range"] = _safe_return_series(sub["high"], sub["low"])
        sub[f"{prefix}_vwap"] = sub[f"{prefix}_amount"] / sub[f"{prefix}_volume"].where(sub[f"{prefix}_volume"] > 0)
        sub[f"{prefix}_pressure"] = sub[f"{prefix}_ret"] * sub[f"{prefix}_amount_share"]
        columns = [
            f"{prefix}_ret",
            f"{prefix}_amount",
            f"{prefix}_volume",
            f"{prefix}_amount_share",
            f"{prefix}_range",
            f"{prefix}_vwap",
            f"{prefix}_pressure",
        ]
        out = out.merge(sub[["symbol", "trade_date", *columns]], on=["symbol", "trade_date"], how="left", suffixes=("", "_overlay"))
        for column in columns:
            overlay = f"{column}_overlay"
            if overlay in out.columns:
                out[column] = out[overlay].where(out[overlay].notna(), out[column])
                out = out.drop(columns=[overlay])
    return out


def _override_open_gap_from_daily(*, root: Path, features: pd.DataFrame, five_rows: pd.DataFrame, memory_limit: str) -> pd.DataFrame:
    if features.empty:
        return features
    active = read_active_manifest(root)
    daily_id = str(dict(active.get("raw", {}) or {}).get("market_daily_raw", "") or "")
    path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    if path is None:
        return features
    daily = read_dataset_manifest(path)
    import duckdb  # type: ignore

    first_open = (
        five_rows.sort_values(["symbol", "trade_date", "bar_time"])
        .groupby(["symbol", "trade_date"], dropna=False)["open"]
        .first()
        .rename("first_open")
        .reset_index()
    )
    keys = features[["symbol", "trade_date"]].drop_duplicates()
    paths = [str(_resolve(root, item.path)) for item in daily.shards]
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.register("keys", keys)
        rows = con.execute(
            """
            with daily as (
              select symbol, trade_date, close,
                     lag(close) over (partition by symbol order by trade_date) as prev_close
              from read_parquet(?, union_by_name=true)
            )
            select k.symbol, k.trade_date, d.prev_close
            from keys k
            left join daily d on d.symbol = k.symbol and d.trade_date = k.trade_date
            """,
            [paths],
        ).fetchdf()
    out = features.merge(rows, on=["symbol", "trade_date"], how="left").merge(first_open, on=["symbol", "trade_date"], how="left")
    if "open_gap" in out.columns:
        out["open_gap"] = _safe_return_series(out["first_open"], out["prev_close"])
    if "open_gap_first_30m_follow_through" in out.columns and "first_30m_ret" in out.columns:
        sign = pd.to_numeric(out["open_gap"], errors="coerce").map(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
        out["open_gap_first_30m_follow_through"] = sign * pd.to_numeric(out["first_30m_ret"], errors="coerce")
    if "open_gap_first_30m_reversal" in out.columns and "first_30m_ret" in out.columns:
        sign = pd.to_numeric(out["open_gap"], errors="coerce").map(lambda value: 1.0 if value > 0 else (-1.0 if value < 0 else 0.0))
        out["open_gap_first_30m_reversal"] = -sign * pd.to_numeric(out["first_30m_ret"], errors="coerce")
    return out.drop(columns=["prev_close", "first_open"], errors="ignore")


def _safe_return_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den > 0) - 1.0


def _rewrite_feature_shard(*, source_path: Path, target_path: Path, remove_keys: pd.DataFrame, replacements: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    data = pd.read_parquet(source_path)
    key_set = {(str(row.symbol), str(row.trade_date)) for row in remove_keys[["symbol", "trade_date"]].drop_duplicates().itertuples(index=False)}
    before = len(data)
    if key_set:
        mask = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]].astype(str)).isin(key_set)
        data = data.loc[~mask].copy()
    removed = before - len(data)
    if not replacements.empty:
        for column in columns:
            if column not in replacements.columns:
                replacements[column] = pd.NA
        data = pd.concat([data, replacements.loc[:, columns]], ignore_index=True)
    data = data.drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target_path, index=False)
    return {"row_count": int(len(data)), "removed_rows": int(removed), "inserted_rows": int(len(replacements))}


def _entry(*, target_dataset_id: str, target_path: Path, source_path: Path, source_shard: ShardManifestEntry, row_count: int) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=f"datasets/intraday_daily_features/{target_dataset_id}/shards/{target_path.name}",
        row_count=int(row_count),
        start_date=source_shard.start_date,
        end_date=source_shard.end_date,
        status="stored",
        file_size=int(target_path.stat().st_size),
        schema_hash=source_shard.schema_hash,
        source_path=str(source_path.resolve()),
        content_key=source_shard.content_key,
        metadata={**dict(source_shard.metadata or {}), "repair": "intraday_zero_features"},
    )


def _manifest(root: Path, dataset_id: str, domain: str) -> Path:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        raise FileNotFoundError(f"dataset manifest not found: {domain} {dataset_id}")
    return path


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _copy_or_link(source_path: Path, target_path: Path, *, mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if str(mode or "").lower() == "copy":
        shutil.copy2(source_path, target_path)
        return
    try:
        os.link(source_path, target_path)
    except Exception:
        shutil.copy2(source_path, target_path)
