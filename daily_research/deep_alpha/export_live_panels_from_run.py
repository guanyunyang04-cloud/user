from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

from daily_research.baseline.data_provider import get_latest_completed_trading_date
import daily_research.deep_alpha.run_deep_alpha_research as research_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh live candidate panels from an existing deep_alpha run without retraining.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing deep_alpha output run directory.")
    parser.add_argument("--end-date", default="", help="Optional latest signal date override, default is latest completed trading date.")
    parser.add_argument("--data-source", choices=["tq", "lake"], default="tq")
    parser.add_argument("--lake-dataset-id", default="")
    parser.add_argument("--data-lake-root", default="")
    return parser.parse_args()


def _normalize_horizon_weights(raw: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except Exception:
            continue
    return out


def _panel_latest_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, usecols=["date"])
    except Exception:
        return None
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def _ordered_feature_frames(
    feature_frames: dict[str, pd.DataFrame],
    expected_feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    missing = [name for name in expected_feature_names if name not in feature_frames]
    if missing:
        raise KeyError(f"Missing expected feature frames: {missing[:10]}")
    return {name: feature_frames[name] for name in expected_feature_names}


def _ordered_target_frames(
    target_frames: dict[str, pd.DataFrame],
    expected_target_names: list[str],
) -> dict[str, pd.DataFrame]:
    missing = [name for name in expected_target_names if name not in target_frames]
    if missing:
        raise KeyError(f"Missing expected target frames: {missing[:10]}")
    return {name: target_frames[name] for name in expected_target_names}


def _expected_style_names(feature_names: list[str]) -> list[str]:
    names: list[str] = []
    for feature_name in feature_names:
        text = str(feature_name)
        if not text.startswith("style_"):
            continue
        suffix = "_strength_20" if text.endswith("_strength_20") else "_member" if text.endswith("_member") else ""
        if not suffix:
            continue
        style_name = text.removeprefix("style_").removesuffix(suffix)
        if style_name and style_name not in names:
            names.append(style_name)
    return names


def _augment_style_map_for_feature_contract(
    style_map: pd.DataFrame | None,
    *,
    columns: list[str],
    expected_feature_names: list[str],
    cache_path: Path | str | None = None,
    keep_existing: bool = True,
) -> pd.DataFrame:
    expected_styles = _expected_style_names(expected_feature_names)
    base = style_map.copy() if style_map is not None and not style_map.empty else pd.DataFrame(index=columns)
    base.index = base.index.astype(str).str.upper()
    base = base.reindex([str(item).upper() for item in columns]).fillna(False).astype(bool)
    if not keep_existing:
        base = base[[name for name in expected_styles if name in base.columns]].copy()
    missing_styles = [name for name in expected_styles if name not in base.columns]
    if missing_styles:
        resolved_cache = Path(cache_path) if cache_path is not None else Path(__file__).resolve().parents[1] / "cache" / "style_map_tq.csv"
        if resolved_cache.exists():
            cached = pd.read_csv(resolved_cache, dtype={"stock": str})
            if "stock" in cached.columns:
                cached["stock"] = cached["stock"].astype(str).str.upper()
                cached = cached.drop_duplicates(subset=["stock"]).set_index("stock")
                for style_name in missing_styles:
                    if style_name in cached.columns:
                        base[style_name] = cached[style_name].reindex(base.index).fillna(False).astype(bool)
    return base


def refresh_live_panels_for_run(
    run_dir: Path,
    latest_end_date: str | None = None,
    *,
    data_source: str = "tq",
    lake_dataset_id: str = "",
    data_lake_root: str = "",
    production_manifest: dict[str, Any] | None = None,
    sector_board_view_id: str = "",
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "deep_alpha_model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"deep_alpha_model.pt not found: {model_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = research_main.DeepAlphaConfig(**artifact["config"])
    latest_signal_date = latest_end_date or pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d")
    cfg.end_date = str(latest_signal_date)

    score_head_method = str(metrics.get("score_head_method", "manual"))
    score_head_artifact = research_main.load_pickle(run_dir / "score_head_artifact.pkl")
    if score_head_method != "manual" and score_head_artifact is None:
        raise RuntimeError(
            f"Run {run_dir.name} uses score_head_method={score_head_method}, but score_head_artifact.pkl is missing."
        )
    risk_gate_artifact = research_main.load_pickle(run_dir / "risk_gate_artifact.pkl")
    score_risk_mode = str(metrics.get("score_risk_mode", "subtract"))
    if score_risk_mode in {"state_gate", "state_liquidity_gate"} and risk_gate_artifact is None:
        raise RuntimeError(
            f"Run {run_dir.name} uses score_risk_mode={score_risk_mode}, but risk_gate_artifact.pkl is missing."
        )

    manifest = production_manifest if isinstance(production_manifest, dict) else {}
    resolved_data_source = str(data_source or manifest.get("data_source", "") or "tq").strip().lower()
    if resolved_data_source not in {"tq", "lake"}:
        resolved_data_source = "tq"
    resolved_lake_dataset_id = str(
        lake_dataset_id
        or manifest.get("lake_dataset_id", "")
        or manifest.get("source_market_dataset_id", "")
        or ""
    ).strip()
    resolved_data_lake_root = str(data_lake_root or manifest.get("data_lake_root", "") or "").strip()
    args = SimpleNamespace(
        data_source=resolved_data_source,
        csv_folder=None,
        stocks_file="" if resolved_data_source == "lake" else str(metrics.get("stocks_file", "") or ""),
        liquidity_pool=str(metrics.get("liquidity_pool", "") or ""),
        rolling_liquidity_pool=str(metrics.get("rolling_liquidity_pool", "") or ""),
        pool_rebalance_days=int(metrics.get("rolling_pool_rebalance_days", 21) or 21),
        pool_adv_window=int(metrics.get("rolling_pool_adv_window", 20) or 20),
        relation_layer=bool(metrics.get("relation_layer", False)),
        use_cache=True,
        refresh_cache=False,
    )

    stocks_file = args.stocks_file or None
    universe = research_main.load_stocks_from_file(stocks_file) if stocks_file else []
    rolling_membership_frame = None
    rolling_pool_key = ""
    if args.data_source == "lake":
        if not resolved_lake_dataset_id:
            raise ValueError("Lake live panel refresh requires lake_dataset_id.")
        lake_payload = research_main.load_lake_market_data_with_pool_view(
            data_lake_root=resolved_data_lake_root,
            lake_dataset_id=resolved_lake_dataset_id,
            sector_board_view_id=str(sector_board_view_id or manifest.get("sector_board_view_id", "") or ""),
            start_date=str(cfg.start_date),
            end_date=str(cfg.end_date),
            benchmark=str(cfg.benchmark),
            pool_name=str(args.rolling_liquidity_pool or args.liquidity_pool or "learned_all_a"),
        )
        raw_df_dict = dict(lake_payload["df_dict"])
        raw_key = str(lake_payload["raw_key"])
        benchmark_open = lake_payload["benchmark_open"]
        benchmark_close = lake_payload["benchmark_close"]
        df_dict = dict(lake_payload["df_dict"])
        rolling_membership_frame = lake_payload["rolling_membership_frame"]
        rolling_pool_key = str(lake_payload.get("rolling_pool_key", "") or "")
    elif args.data_source == "tq":
        if args.rolling_liquidity_pool:
            try:
                universe = research_main.load_universe_from_tq(cfg.universe_scope)
            except Exception:
                universe = research_main._load_cached_rolling_pool_union(
                    pool_name=args.rolling_liquidity_pool,
                    start_date=cfg.start_date,
                    end_date=cfg.end_date,
                )
                if not universe:
                    raise
        elif not universe and cfg.universe_scope == "all_a":
            universe = research_main.load_universe_from_tq(cfg.universe_scope)
        elif not universe:
            raise ValueError("TQ mode without stocks requires a valid stocks_file or universe_scope=all_a.")

        raw_df_dict, raw_key = research_main.load_raw_market_data(
            cfg,
            args,
            universe,
            progress_desc="Refresh live candidate market data",
        )
        benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
        df_dict, benchmark_close = research_main.split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    else:
        raise ValueError(f"Unsupported data_source for live panel refresh: {args.data_source}")

    rolling_pool_artifact = None
    if args.data_source != "lake" and args.rolling_liquidity_pool:
        rolling_pool_artifact, rolling_pool_key = research_main.load_cached_or_build_rolling_pool(
            args=args,
            cfg=cfg,
            df_dict=df_dict,
            raw_key=raw_key,
        )
        rolling_membership_frame = rolling_pool_artifact.membership_frame
        rolling_union = rolling_membership_frame.columns[rolling_membership_frame.any(axis=0)].tolist()
        if not rolling_union:
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} research pool is empty for the requested window.")
        df_dict = research_main.subset_df_dict_to_stocks(df_dict, rolling_union)
        rolling_membership_frame = (
            rolling_membership_frame
            .reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns)
            .fillna(False)
        )

    close = df_dict["Close"]
    valid_start = pd.Timestamp(metrics.get("valid_start") or artifact.get("valid_start"))
    if pd.isna(valid_start):
        raise RuntimeError(f"Run {run_dir.name} is missing valid_start metadata.")

    state_frame, _state_name_map, state_key = research_main.load_cached_or_fit_market_state(
        args=args,
        cfg=cfg,
        benchmark_close=benchmark_close,
        raw_key=raw_key,
    )
    liquidity_bucket_frame, liquidity_bucket_key = research_main.load_cached_or_build_liquidity_buckets(
        args=args,
        cfg=cfg,
        amount_frame=df_dict["Amount"],
        raw_key=raw_key,
    )

    industry_map = None
    style_map = None
    if (args.relation_layer or cfg.dynamic_graph_layer) and args.data_source == "lake":
        industry_map = lake_payload.get("industry_map")
        style_map = lake_payload.get("style_map")
    if (args.relation_layer or cfg.dynamic_graph_layer) and args.data_source == "tq":
        try:
            industry_map = research_main.load_industry_map_from_tq(list(close.columns))
        except Exception:
            industry_map = None
        try:
            style_map = research_main.load_style_map_from_tq(list(close.columns))
        except Exception:
            style_map = None

    target_names = list(artifact["target_names"])
    feature_names = list(artifact["feature_names"])
    if (args.relation_layer or cfg.dynamic_graph_layer) and args.data_source == "lake":
        style_map = _augment_style_map_for_feature_contract(
            style_map,
            columns=list(close.columns),
            expected_feature_names=feature_names,
            keep_existing=False,
        )
    breakout_event_task = any(str(name).startswith("event_breakout_") for name in target_names)
    clean_breakout_event_task = any(str(name).startswith("event_clean_breakout_") for name in target_names)

    feature_meta = {
        "version": 4,
        "raw_key": raw_key,
        "relation_layer": bool(args.relation_layer),
        "liquidity_layer": bool(cfg.liquidity_layer),
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "dynamic_graph_layer": bool(cfg.dynamic_graph_layer),
        "dynamic_graph_top_k": int(cfg.dynamic_graph_top_k),
        "dynamic_graph_temperature": float(cfg.dynamic_graph_temperature),
        "dynamic_graph_industry_boost": float(cfg.dynamic_graph_industry_boost),
        "dynamic_graph_style_boost": float(cfg.dynamic_graph_style_boost),
        "short_alpha_features": bool(cfg.short_alpha_features),
        "prediction_horizons": list(cfg.prediction_horizons),
        "breakout_event_horizon": int(cfg.breakout_event_horizon),
        "breakout_event_threshold": float(cfg.breakout_event_threshold),
        "breakout_event_pullback_limit": float(cfg.breakout_event_pullback_limit),
        "breakout_event_task": bool(breakout_event_task),
        "clean_breakout_event_task": bool(clean_breakout_event_task),
        "market_state_count": cfg.market_state_count,
        "state_key": state_key,
        "liquidity_bucket_key": liquidity_bucket_key,
        "industry_signature": research_main.series_signature(industry_map),
        "style_signature": research_main.frame_signature(style_map),
        "rolling_pool_key": rolling_pool_key,
    }
    feature_key = research_main.cache_key(feature_meta)
    feature_path = research_main.get_cache_root() / "features" / f"{feature_key}.pkl"
    feature_cached = research_main.load_pickle(feature_path)
    if feature_cached is not None:
        feature_frames = feature_cached["feature_frames"]
        target_frames = feature_cached["target_frames"]
        structure_label_frame = feature_cached.get("structure_label_frame")
    else:
        feature_frames = research_main.build_sequence_features(
            df_dict,
            benchmark_close,
            state_frame,
            industry_map=industry_map,
            style_map=style_map,
            liquidity_layer=cfg.liquidity_layer,
            liquidity_bucket_count=cfg.liquidity_bucket_count,
            liquidity_bucket_frame=liquidity_bucket_frame,
            dynamic_graph_layer=cfg.dynamic_graph_layer,
            dynamic_graph_top_k=cfg.dynamic_graph_top_k,
            dynamic_graph_temperature=cfg.dynamic_graph_temperature,
            dynamic_graph_industry_boost=cfg.dynamic_graph_industry_boost,
            dynamic_graph_style_boost=cfg.dynamic_graph_style_boost,
            short_alpha_features=cfg.short_alpha_features,
        )
        target_frames = research_main.build_targets(
            close,
            benchmark_close,
            cfg.prediction_horizons,
            open_df=df_dict["Open"],
            benchmark_open=benchmark_open,
            execution_mode="next_open",
            breakout_event_horizon=cfg.breakout_event_horizon,
            breakout_event_threshold=cfg.breakout_event_threshold,
            breakout_event_pullback_limit=cfg.breakout_event_pullback_limit,
            breakout_event_task=breakout_event_task,
            clean_breakout_event_task=clean_breakout_event_task,
        )
        structure_label_frame = research_main.build_structure_label_frame(
            close=df_dict["Close"].astype(float),
            open_df=df_dict["Open"].astype(float),
            high=df_dict["High"].astype(float),
            low=df_dict["Low"].astype(float),
            amount=df_dict["Amount"].astype(float),
            benchmark_close=benchmark_close.astype(float).reindex(df_dict["Close"].index),
        )
        research_main.save_pickle(
            feature_path,
            {
                "feature_frames": feature_frames,
                "target_frames": target_frames,
                "structure_label_frame": structure_label_frame,
            },
        )
    feature_frames = _ordered_feature_frames(feature_frames, feature_names)
    target_frames = _ordered_target_frames(target_frames, target_names)
    if structure_label_frame is None:
        structure_label_frame = research_main.build_structure_label_frame(
            close=df_dict["Close"].astype(float),
            open_df=df_dict["Open"].astype(float),
            high=df_dict["High"].astype(float),
            low=df_dict["Low"].astype(float),
            amount=df_dict["Amount"].astype(float),
            benchmark_close=benchmark_close.astype(float).reindex(df_dict["Close"].index),
        )

    train_target_frames = research_main.transform_return_target_frames(target_frames, cfg.return_target_transform)
    train_target_frames = _ordered_target_frames(train_target_frames, target_names)
    model = research_main.MultiTaskRanker(
        input_dim=len(feature_names),
        hidden_dim=cfg.hidden_dim,
        output_dim=len(target_names),
        return_output_dim=sum(1 for name in target_names if name.startswith("fwd_excess_")),
        risk_output_dim=sum(1 for name in target_names if not name.startswith("fwd_excess_")),
        dropout=cfg.dropout,
        encoder_family=cfg.encoder_family,
        patch_len=cfg.patch_len,
        return_head_mode=cfg.return_head_mode,
        context_dim=cfg.context_dim,
        state_context=cfg.state_context,
        liquidity_context=cfg.liquidity_context,
        structure_context=cfg.structure_context,
        aux_structure_task=cfg.aux_structure_task,
        structure_prototype_task=cfg.structure_prototype_task,
        state_vocab_size=cfg.market_state_count,
        liquidity_bucket_count=cfg.liquidity_bucket_count,
        structure_vocab_size=len(research_main.STRUCTURE_LABELS),
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
    )
    model.load_state_dict(artifact["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_profile = research_main.resolve_runtime_profile(
        stage="finetune",
        encoder_family=cfg.encoder_family,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        use_amp=cfg.use_amp,
        safe_profile=bool(cfg.safe_runtime_profile),
    )
    cfg.batch_size = runtime_profile.batch_size
    cfg.num_workers = runtime_profile.num_workers
    cfg.pin_memory = runtime_profile.pin_memory
    cfg.use_amp = runtime_profile.use_amp
    research_main.configure_torch_runtime(device, use_amp=bool(cfg.use_amp))
    model = model.to(device)

    pin_memory = bool(cfg.pin_memory and torch.cuda.is_available())
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(cfg.num_workers),
        "collate_fn": research_main.collate_batch,
        "pin_memory": pin_memory,
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        if runtime_profile.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(runtime_profile.prefetch_factor)

    live_outputs = research_main._build_live_inference_outputs(
        cfg=cfg,
        close=close,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        open_df=df_dict["Open"],
        amount_frame=df_dict["Amount"],
        feature_frames=feature_frames,
        train_target_frames=train_target_frames,
        target_frames=target_frames,
        state_frame=state_frame,
        liquidity_bucket_frame=liquidity_bucket_frame,
        structure_label_frame=structure_label_frame,
        rolling_membership_frame=rolling_membership_frame,
        live_start=valid_start,
        model=model,
        loader_kwargs=loader_kwargs,
        device=device,
        target_names=target_names,
        score_head_method=score_head_method,
        score_head_artifact=score_head_artifact,
        applied_score_horizon_weights=_normalize_horizon_weights(metrics.get("applied_score_horizon_weights")),
        applied_score_downside_penalty=float(metrics.get("applied_score_downside_penalty", metrics.get("score_downside_penalty", 0.0))),
        risk_gate_artifact=risk_gate_artifact,
        use_amp=bool(cfg.use_amp),
    )
    live_score_frame = live_outputs["score_frame"]
    live_target_weights = live_outputs["target_weights"]
    portfolio_capped_live_target_weights = live_outputs["portfolio_capped_target_weights"]
    live_latest_scores = None
    if not live_score_frame.dropna(how="all").empty:
        live_latest_scores = live_score_frame.loc[[live_score_frame.dropna(how="all").index.max()]].T.reset_index()
        live_latest_scores.columns = ["stock", "latest_score"]
        live_latest_scores = live_latest_scores.sort_values("latest_score", ascending=False, na_position="last")
    research_main._panel_to_long(live_score_frame, "score").to_csv(
        run_dir / "daily_live_score_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    research_main._panel_to_long(live_target_weights, "target_weight").to_csv(
        run_dir / "daily_live_target_weight_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    research_main._panel_to_long(portfolio_capped_live_target_weights, "target_weight").to_csv(
        run_dir / "portfolio_capped_daily_live_target_weight_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if live_latest_scores is not None:
        live_latest_scores.to_csv(run_dir / "live_latest_scores.csv", index=False, encoding="utf-8-sig")

    execution_aligned_profile = str(metrics.get("execution_alignment_profile", "") or "")
    execution_aligned_profile_spec = (
        metrics.get("execution_alignment_selected_profile_spec")
        if isinstance(metrics.get("execution_alignment_selected_profile_spec"), dict)
        else {}
    )
    if execution_aligned_profile:
        execution_aligned_live_outputs = research_main._build_live_execution_aligned_outputs(
            execution_alignment_artifact=SimpleNamespace(
                selected_profile=execution_aligned_profile,
                selected_profile_spec=execution_aligned_profile_spec,
            ),
            raw_live_score_frame=live_score_frame,
            raw_live_target_weights=live_target_weights,
            close=close,
            benchmark_close=benchmark_close,
            open_df=df_dict["Open"],
            benchmark_open=benchmark_open,
            cfg=cfg,
            transaction_cost_bps=float(metrics.get("execution_alignment_transaction_cost_bps", 0.0)),
            slippage_bps=float(metrics.get("execution_alignment_slippage_bps", 0.0)),
            sell_tax_bps=float(metrics.get("execution_alignment_sell_tax_bps", 0.0)),
        )
        if execution_aligned_live_outputs is not None:
            aligned_score_frame = execution_aligned_live_outputs["score_frame"]
            aligned_target_weights = execution_aligned_live_outputs["target_weights"]
            research_main._panel_to_long(aligned_score_frame, "score").to_csv(
                run_dir / "execution_aligned_daily_live_score_panel.csv",
                index=False,
                encoding="utf-8-sig",
            )
            research_main._panel_to_long(aligned_target_weights, "target_weight").to_csv(
                run_dir / "execution_aligned_daily_live_target_weight_panel.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if not aligned_score_frame.dropna(how="all").empty:
                aligned_latest_scores = aligned_score_frame.loc[[aligned_score_frame.dropna(how="all").index.max()]].T.reset_index()
                aligned_latest_scores.columns = ["stock", "latest_score"]
                aligned_latest_scores = aligned_latest_scores.sort_values("latest_score", ascending=False, na_position="last")
                aligned_latest_scores.to_csv(run_dir / "execution_aligned_live_latest_scores.csv", index=False, encoding="utf-8-sig")

    live_signal_date = _panel_latest_date(run_dir / "daily_live_target_weight_panel.csv")
    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "data_source": args.data_source,
        "lake_dataset_id": resolved_lake_dataset_id,
        "data_lake_root": resolved_data_lake_root,
        "sector_board_view_id": str(sector_board_view_id or manifest.get("sector_board_view_id", "") or ""),
        "live_signal_date": "" if live_signal_date is None else str(live_signal_date.date()),
        "latest_end_date": str(pd.Timestamp(cfg.end_date).date()),
        "target_weight_panel": str((run_dir / "daily_live_target_weight_panel.csv").resolve()),
        "score_panel": str((run_dir / "daily_live_score_panel.csv").resolve()),
        "execution_aligned_target_weight_panel": str((run_dir / "execution_aligned_daily_live_target_weight_panel.csv").resolve()),
        "execution_aligned_score_panel": str((run_dir / "execution_aligned_daily_live_score_panel.csv").resolve()),
        "raw_target_panel_rows": int(len(live_target_weights.stack(dropna=False))),
        "raw_score_panel_rows": int(len(live_score_frame.stack(dropna=False))),
    }
    aligned_target_path = run_dir / "execution_aligned_daily_live_target_weight_panel.csv"
    aligned_score_path = run_dir / "execution_aligned_daily_live_score_panel.csv"
    if aligned_target_path.exists():
        try:
            aligned_target = pd.read_csv(aligned_target_path)
            summary["execution_aligned_target_panel_rows"] = int(len(aligned_target))
            if "target_weight" in aligned_target.columns and "date" in aligned_target.columns:
                latest_aligned = pd.to_datetime(aligned_target["date"], errors="coerce")
                latest_date = latest_aligned.max()
                summary["execution_aligned_signal_date"] = "" if pd.isna(latest_date) else str(pd.Timestamp(latest_date).date())
                latest_rows = aligned_target.loc[latest_aligned.eq(latest_date)] if not pd.isna(latest_date) else aligned_target.iloc[0:0]
                weights = pd.to_numeric(latest_rows.get("target_weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
                summary["target_position_count"] = int((weights > 0.0).sum())
        except Exception:
            pass
    if aligned_score_path.exists():
        try:
            aligned_score = pd.read_csv(aligned_score_path)
            summary["execution_aligned_score_panel_rows"] = int(len(aligned_score))
        except Exception:
            pass
    manifest_path = run_dir / "live_panel_refresh_manifest.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest_path"] = str(manifest_path.resolve())
    return summary


def main() -> None:
    args = parse_args()
    summary = refresh_live_panels_for_run(
        Path(args.run_dir),
        latest_end_date=args.end_date or None,
        data_source=args.data_source,
        lake_dataset_id=args.lake_dataset_id,
        data_lake_root=args.data_lake_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
