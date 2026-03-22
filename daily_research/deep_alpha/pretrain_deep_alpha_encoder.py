from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.deep_alpha.cache_utils import cache_key, frame_signature, get_cache_root, load_pickle, save_pickle, series_signature
from daily_research.deep_alpha.config import DeepAlphaConfig
from daily_research.deep_alpha.models import MaskedPatchPretrainer
from daily_research.deep_alpha.pipeline_utils import (
    build_target_loss_weights,
    load_cached_or_build_liquidity_buckets,
    load_cached_or_build_rolling_pool,
    load_cached_or_fit_market_state,
    load_raw_market_data,
    load_stocks_from_file,
    parse_horizons,
    parse_stocks,
    resolve_split_dates,
    resolve_stocks_file,
    subset_df_dict_to_stocks,
)
from daily_research.deep_alpha.runtime_profile import configure_torch_runtime, resolve_runtime_profile
from daily_research.deep_alpha.sequence_dataset import (
    SequenceOnlyDataset,
    build_sequence_corpus,
    build_sequence_features,
    build_structure_label_frame,
    build_targets,
)
from daily_research.deep_alpha.trainer import collate_pretrain_batch, train_masked_pretrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Patch-based masked self-supervised pretraining for deep_alpha encoder")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--liquidity-pool", choices=["liquid300", "liquid500", "liquid800"], default="")
    parser.add_argument("--rolling-liquidity-pool", choices=["liquid300", "liquid500", "liquid800"], default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--lookback-window", type=int, default=120)
    parser.add_argument("--prediction-horizons", default="5,10,20")
    parser.add_argument("--train-end-date", default="")
    parser.add_argument("--valid-start-date", default="")
    parser.add_argument("--valid-days", type=int, default=252, help="Downstream formal holdout size; pretraining never uses dates after the implied train_end.")
    parser.add_argument("--pretrain-valid-days", type=int, default=63, help="Last N train-side trading days reserved for self-supervised validation.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=5)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-improvement", type=float, default=1e-4)
    parser.add_argument("--auto-extend-undertrained", action="store_true", help="Auto-extend pretraining when diagnostics still show undertrained at the current cap.")
    parser.add_argument("--no-auto-extend-undertrained", dest="auto_extend_undertrained", action="store_false")
    parser.add_argument("--epoch-extend-step", type=int, default=4, help="Epochs added each time auto-extension triggers.")
    parser.add_argument("--max-total-epochs", type=int, default=20, help="Hard ceiling for adaptive pretraining extension.")
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--safe-runtime-profile", action="store_true", help="Auto-cap batch size and workers for this local machine.")
    parser.add_argument("--no-safe-runtime-profile", dest="safe_runtime_profile", action="store_false")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--market-state-count", type=int, default=4)
    parser.add_argument("--liquidity-layer", action="store_true")
    parser.add_argument("--liquidity-bucket-count", type=int, default=5)
    parser.add_argument("--relation-layer", action="store_true")
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--experiment-tag", default="")
    parser.set_defaults(
        pin_memory=True,
        use_amp=True,
        use_cache=True,
        safe_runtime_profile=True,
        auto_extend_undertrained=True,
    )
    return parser.parse_args()


def _resolve_pretrain_split(train_dates: pd.Index, pretrain_valid_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(train_dates)
    if len(dates) <= max(pretrain_valid_days + 1, 5):
        raise RuntimeError("Not enough train-side dates for pretraining split.")
    valid_start = dates[max(0, len(dates) - int(pretrain_valid_days))]
    train_end = dates[dates.get_loc(valid_start) - 1]
    return pd.Timestamp(train_end), pd.Timestamp(valid_start)


def main():
    args = parse_args()
    if args.liquidity_pool and args.rolling_liquidity_pool:
        raise ValueError("Use either --liquidity-pool or --rolling-liquidity-pool, not both.")
    if args.rolling_liquidity_pool and (args.stocks or args.stocks_file):
        raise ValueError("Use either a fixed --stocks/--stocks-file universe or --rolling-liquidity-pool, not both.")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    horizons = parse_horizons(args.prediction_horizons)
    cfg = DeepAlphaConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        universe_scope=args.universe_scope,
        lookback_window=args.lookback_window,
        prediction_horizons=horizons,
        train_end_date=args.train_end_date,
        valid_start_date=args.valid_start_date,
        valid_days=args.valid_days,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        hidden_dim=args.hidden_dim,
        encoder_family="patch_transformer",
        patch_len=args.patch_len,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        early_stop_patience=args.early_stop_patience,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_factor=args.lr_plateau_factor,
        min_improvement=args.min_improvement,
        use_amp=args.use_amp,
        safe_runtime_profile=args.safe_runtime_profile,
        liquidity_layer=args.liquidity_layer,
        liquidity_bucket_count=args.liquidity_bucket_count,
        random_seed=args.random_seed,
        market_state_count=args.market_state_count,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        target_loss_weights=build_target_loss_weights(horizons, ""),
    )
    if not cfg.end_date:
        cfg.end_date = pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d")

    stocks_file = resolve_stocks_file(args)
    universe = load_stocks_from_file(stocks_file) or parse_stocks(args.stocks)
    if args.data_source == "tq":
        if args.rolling_liquidity_pool:
            print(f"[1/7] Loading base universe for rolling {args.rolling_liquidity_pool} pretraining pool...")
            universe = load_universe_from_tq(cfg.universe_scope)
        elif args.liquidity_pool and not universe:
            from daily_research.execution.liquidity_universe import get_named_pool_file

            pool_file = get_named_pool_file(args.liquidity_pool)
            universe = load_stocks_from_file(str(pool_file))
        elif not universe and cfg.universe_scope == "all_a":
            print("[1/7] Loading all-A universe from TQ...")
            universe = load_universe_from_tq(cfg.universe_scope)
        elif not universe:
            raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")

    raw_df_dict, raw_key = load_raw_market_data(cfg, args, universe)
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)

    rolling_pool_artifact = None
    rolling_membership_frame = None
    rolling_pool_key = ""
    if args.rolling_liquidity_pool:
        rolling_pool_artifact, rolling_pool_key = load_cached_or_build_rolling_pool(
            args=args,
            cfg=cfg,
            df_dict=df_dict,
            raw_key=raw_key,
        )
        rolling_membership_frame = rolling_pool_artifact.membership_frame
        rolling_union = rolling_membership_frame.columns[rolling_membership_frame.any(axis=0)].tolist()
        if not rolling_union:
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} pretraining pool is empty for the requested window.")
        df_dict = subset_df_dict_to_stocks(df_dict, rolling_union)
        rolling_membership_frame = rolling_membership_frame.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns).fillna(False)

    close = df_dict["Close"]
    train_end, valid_start = resolve_split_dates(close.index, cfg.train_end_date, cfg.valid_start_date, cfg.valid_days)
    pretrain_train_end, pretrain_valid_start = _resolve_pretrain_split(close.index[close.index <= train_end], args.pretrain_valid_days)

    state_frame, _state_name_map, state_key = load_cached_or_fit_market_state(
        args=args,
        cfg=cfg,
        benchmark_close=benchmark_close,
        raw_key=raw_key,
    )
    liquidity_bucket_frame, liquidity_bucket_key = load_cached_or_build_liquidity_buckets(
        args=args,
        cfg=cfg,
        amount_frame=df_dict["Amount"],
        raw_key=raw_key,
    )

    print("[4/7] Building features for masked pretraining...")
    industry_map = None
    style_map = None
    if args.relation_layer and args.data_source == "tq":
        try:
            industry_map = load_industry_map_from_tq(list(close.columns))
        except Exception as exc:
            print(f"      Industry map unavailable: {exc}")
        try:
            style_map = load_style_map_from_tq(list(close.columns))
        except Exception as exc:
            print(f"      Style map unavailable: {exc}")

    feature_meta = {
        "version": 1,
        "mode": "masked_pretrain",
        "raw_key": raw_key,
        "relation_layer": bool(args.relation_layer),
        "liquidity_layer": bool(cfg.liquidity_layer),
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "state_key": state_key,
        "liquidity_bucket_key": liquidity_bucket_key,
        "industry_signature": series_signature(industry_map),
        "style_signature": frame_signature(style_map),
        "rolling_pool_key": rolling_pool_key,
    }
    feature_key = cache_key(feature_meta)
    feature_path = get_cache_root() / "features" / f"{feature_key}.pkl"
    feature_cached = load_pickle(feature_path) if args.use_cache and not args.refresh_cache else None
    if feature_cached is not None:
        print(f"      Loading feature cache: {feature_path.name}")
        feature_frames = feature_cached["feature_frames"]
        target_frames = feature_cached["target_frames"]
        structure_label_frame = feature_cached["structure_label_frame"]
    else:
        feature_frames = build_sequence_features(
            df_dict,
            benchmark_close,
            state_frame,
            industry_map=industry_map,
            style_map=style_map,
            liquidity_layer=cfg.liquidity_layer,
            liquidity_bucket_count=cfg.liquidity_bucket_count,
            liquidity_bucket_frame=liquidity_bucket_frame,
        )
        target_frames = build_targets(
            close,
            benchmark_close,
            cfg.prediction_horizons,
            open_df=df_dict["Open"],
            benchmark_open=benchmark_open,
            execution_mode="next_open",
        )
        structure_label_frame = build_structure_label_frame(
            close=df_dict["Close"].astype(float),
            open_df=df_dict["Open"].astype(float),
            high=df_dict["High"].astype(float),
            low=df_dict["Low"].astype(float),
            amount=df_dict["Amount"].astype(float),
            benchmark_close=benchmark_close.astype(float).reindex(df_dict["Close"].index),
        )
        if args.use_cache:
            save_pickle(
                feature_path,
                {
                    "feature_frames": feature_frames,
                    "target_frames": target_frames,
                    "structure_label_frame": structure_label_frame,
                },
            )
            print(f"      Saved feature cache: {feature_path}")

    pretrain_dates = list(close.index[close.index <= train_end])
    corpus_meta = {
        "version": 1,
        "mode": "masked_pretrain",
        "feature_key": feature_key,
        "lookback_window": int(cfg.lookback_window),
        "min_adv20": float(cfg.min_adv20),
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
        "pretrain_train_end": str(pd.Timestamp(pretrain_train_end).date()),
        "pretrain_valid_start": str(pd.Timestamp(pretrain_valid_start).date()),
        "rolling_pool_key": rolling_pool_key,
    }
    corpus_key = cache_key(corpus_meta)
    corpus_path = get_cache_root() / "corpus" / f"{corpus_key}.pkl"
    corpus = load_pickle(corpus_path) if args.use_cache and not args.refresh_cache else None
    if corpus is not None:
        print(f"      Loading sequence corpus cache: {corpus_path.name}")
    else:
        print("      Building pretraining corpus...")
        corpus = build_sequence_corpus(
            feature_frames=feature_frames,
            train_target_frames=target_frames,
            raw_target_frames=target_frames,
            lookback_window=cfg.lookback_window,
            sample_dates=pretrain_dates,
            min_adv20=cfg.min_adv20,
            amount_frame=df_dict["Amount"],
            close_frame=close,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
            state_frame=state_frame,
            liquidity_bucket_frame=liquidity_bucket_frame,
            structure_label_frame=structure_label_frame,
            universe_membership_frame=rolling_membership_frame,
            require_targets=False,
        )
        if args.use_cache:
            save_pickle(corpus_path, corpus)
            print(f"      Saved sequence corpus cache: {corpus_path}")

    train_ds = SequenceOnlyDataset(corpus=corpus, indices=corpus.build_index(end_date=pretrain_train_end))
    valid_ds = SequenceOnlyDataset(corpus=corpus, indices=corpus.build_index(start_date=pretrain_valid_start, end_date=train_end))
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise RuntimeError("Masked pretraining dataset is empty. Try a longer history or smaller lookback window.")

    print(f"[5/7] Pretraining samples={len(train_ds)} validation samples={len(valid_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_profile = resolve_runtime_profile(
        stage="pretrain",
        encoder_family="patch_transformer",
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
    configure_torch_runtime(device, use_amp=bool(cfg.use_amp))
    for note in runtime_profile.applied_notes:
        print(f"      Runtime tuning: {note}")
    pin_memory = bool(cfg.pin_memory and torch.cuda.is_available())
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "num_workers": int(cfg.num_workers),
        "collate_fn": collate_pretrain_batch,
        "pin_memory": pin_memory,
    }
    if int(cfg.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        if runtime_profile.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(runtime_profile.prefetch_factor)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": int(cfg.num_workers),
        "collate_fn": collate_pretrain_batch,
        "pin_memory": pin_memory,
        "persistent_workers": bool(int(cfg.num_workers) > 0),
    }
    if int(cfg.num_workers) > 0 and runtime_profile.prefetch_factor is not None:
        valid_loader_kwargs["prefetch_factor"] = int(runtime_profile.prefetch_factor)
    valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)

    model = MaskedPatchPretrainer(
        input_dim=len(train_ds.feature_names),
        hidden_dim=cfg.hidden_dim,
        patch_len=cfg.patch_len,
        n_heads=cfg.transformer_heads,
        n_layers=cfg.transformer_layers,
        dropout=cfg.dropout,
        mask_ratio=args.mask_ratio,
    )

    print("[6/7] Training masked patch pretrainer...")
    pretrain_result = train_masked_pretrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        device=device,
        use_amp=bool(cfg.use_amp),
        min_epochs=cfg.min_epochs,
        early_stop_patience=cfg.early_stop_patience,
        lr_plateau_patience=cfg.lr_plateau_patience,
        lr_plateau_factor=cfg.lr_plateau_factor,
        min_improvement=cfg.min_improvement,
        auto_extend_undertrained=bool(args.auto_extend_undertrained),
        epoch_extend_step=int(args.epoch_extend_step),
        max_total_epochs=int(args.max_total_epochs),
    )
    history = pretrain_result.history
    training_diagnostics = pretrain_result.diagnostics
    print(
        f"      Pretraining diagnostics: status={training_diagnostics.status}, "
        f"best_epoch={training_diagnostics.best_epoch}/{training_diagnostics.epochs_completed}, "
        f"best_valid_loss={training_diagnostics.best_valid_loss:.6f}"
    )

    print("[7/7] Writing pretraining artifacts...")
    output_root = Path("daily_research/output")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or f"deep_alpha_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame([record.__dict__ for record in history])
    history_df.to_csv(run_dir / "pretrain_history.csv", index=False, encoding="utf-8-sig")

    artifact = {
        "framework": "deep_alpha_masked_patch_pretrain",
        "encoder_state_dict": model.encoder.state_dict(),
        "feature_names": train_ds.feature_names,
        "input_dim": len(train_ds.feature_names),
        "hidden_dim": cfg.hidden_dim,
        "patch_len": cfg.patch_len,
        "transformer_heads": cfg.transformer_heads,
        "transformer_layers": cfg.transformer_layers,
        "dropout": cfg.dropout,
        "mask_ratio": args.mask_ratio,
        "train_end": str(train_end.date()),
        "pretrain_train_end": str(pretrain_train_end.date()),
        "pretrain_valid_start": str(pretrain_valid_start.date()),
        "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
        "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
        "rolling_pool_adv_window": int(args.pool_adv_window),
        "execution_mode": "next_open",
        "raw_cache_key": raw_key,
        "rolling_pool_cache_key": rolling_pool_key,
        "state_cache_key": state_key,
        "liquidity_bucket_cache_key": liquidity_bucket_key,
        "feature_cache_key": feature_key,
        "corpus_cache_key": corpus_key,
    }
    torch.save(artifact, run_dir / "pretrained_encoder.pt")
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **{k: v for k, v in artifact.items() if not k.endswith("_state_dict")},
                "device": str(device),
                "train_samples": len(train_ds),
                "valid_samples": len(valid_ds),
                "final_train_loss": None if history_df.empty else float(history_df["train_loss"].iloc[-1]),
                "final_valid_loss": None if history_df.empty else float(history_df["valid_loss"].iloc[-1]),
                "epochs": cfg.epochs,
                "min_epochs": cfg.min_epochs,
                "early_stop_patience": cfg.early_stop_patience,
                "lr_plateau_patience": cfg.lr_plateau_patience,
                "lr_plateau_factor": cfg.lr_plateau_factor,
                "min_improvement": cfg.min_improvement,
                "auto_extend_undertrained": bool(args.auto_extend_undertrained),
                "epoch_extend_step": int(args.epoch_extend_step),
                "max_total_epochs": int(args.max_total_epochs),
                "safe_runtime_profile": bool(cfg.safe_runtime_profile),
                "runtime_profile": runtime_profile.__dict__,
                "training_diagnostics": training_diagnostics.__dict__,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Output: {run_dir}")
    print(json.dumps({"final_valid_loss": None if history_df.empty else float(history_df['valid_loss'].iloc[-1])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
