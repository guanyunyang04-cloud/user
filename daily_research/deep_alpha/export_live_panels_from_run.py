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


def refresh_live_panels_for_run(run_dir: Path, latest_end_date: str | None = None) -> dict[str, Any]:
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

    args = SimpleNamespace(
        data_source="tq",
        csv_folder=None,
        stocks_file=str(metrics.get("stocks_file", "") or ""),
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
    if args.data_source == "tq":
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
        progress_desc="刷新 live 候选行情",
    )
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = research_main.split_benchmark_from_universe(raw_df_dict, cfg.benchmark)

    rolling_pool_artifact = None
    rolling_membership_frame = None
    rolling_pool_key = ""
    if args.rolling_liquidity_pool:
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
    if (args.relation_layer or cfg.dynamic_graph_layer) and args.data_source == "tq":
        try:
            industry_map = research_main.load_industry_map_from_tq(list(close.columns))
        except Exception:
            industry_map = None
        try:
            style_map = research_main.load_style_map_from_tq(list(close.columns))
        except Exception:
            style_map = None

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
        "prediction_horizons": list(cfg.prediction_horizons),
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
        )
        target_frames = research_main.build_targets(
            close,
            benchmark_close,
            cfg.prediction_horizons,
            open_df=df_dict["Open"],
            benchmark_open=benchmark_open,
            execution_mode="next_open",
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
    target_names = list(artifact["target_names"])
    model = research_main.MultiTaskRanker(
        input_dim=len(artifact["feature_names"]),
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
    if live_latest_scores is not None:
        live_latest_scores.to_csv(run_dir / "live_latest_scores.csv", index=False, encoding="utf-8-sig")

    execution_aligned_profile = str(metrics.get("execution_alignment_profile", "") or "")
    if execution_aligned_profile:
        execution_aligned_live_outputs = research_main._build_live_execution_aligned_outputs(
            execution_alignment_artifact=SimpleNamespace(selected_profile=execution_aligned_profile),
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
        "run_dir": str(run_dir),
        "live_signal_date": "" if live_signal_date is None else str(live_signal_date.date()),
        "latest_end_date": str(pd.Timestamp(cfg.end_date).date()),
        "target_weight_panel": str((run_dir / "daily_live_target_weight_panel.csv").resolve()),
        "score_panel": str((run_dir / "daily_live_score_panel.csv").resolve()),
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = refresh_live_panels_for_run(Path(args.run_dir), latest_end_date=args.end_date or None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
