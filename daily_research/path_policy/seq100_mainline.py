from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from daily_research.path_policy.qdp_v2_sequence_path_training import (
    DEFAULT_SEED,
    EVALUATION_MODE_STANDARD,
    EVALUATION_MODES,
    INPUT_CHANNEL_PROFILE_ALL,
    INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
    INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
    PATH_LOSS_PROFILE_DEFAULT,
    PATH_LOSS_PROFILE_OHLCVA_EQUAL,
    PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
    PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    PATH_VALUE_DEFAULT_SEMANTIC,
    RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
    RANK_TRAINING_PROFILE_LOCAL_CHUNK,
    SUMMARY_LOSS_PROFILE_BASE,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    main as sequence_training_main,
)

DEFAULT_STORE_VIEW = Path("daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json")
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_DAILY_ONLY_SUMMARY_V2_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2"
LEGACY_ALL_CHANNELS_BASE_RUN_TAG = "seq100_todayclose_path_only_all_channels_base"
DEFAULT_SUMMARY_V2_RUN_TAG = "seq100_todayclose_path_only_summary_v2"
DEFAULT_SUMMARY_V2_NO60_RUN_TAG = "seq100_todayclose_path_only_summary_v2_no60"
DEFAULT_SUMMARY_V2_PRICE_DELTA_RUN_TAG = "seq100_todayclose_path_only_summary_v2_price_delta"
DEFAULT_SUMMARY_V2_OHLCVA_AUX_RUN_TAG = "seq100_todayclose_path_only_summary_v2_ohlcva_aux"
DEFAULT_SUMMARY_V2_OHLCVA_AUX_LOW_RUN_TAG = "seq100_todayclose_path_only_summary_v2_ohlcva_aux_low"
DEFAULT_SUMMARY_V2_OHLCVA_AUX_LOW_PRICE_DELTA_RUN_TAG = (
    "seq100_todayclose_path_only_summary_v2_ohlcva_aux_low_price_delta"
)
DEFAULT_SUMMARY_V2_OHLCVA_PATH_EQUAL_RUN_TAG = "seq100_todayclose_path_only_summary_v2_ohlcva_path_equal"
DEFAULT_DAILY_ONLY_RUN_TAG = "seq100_todayclose_path_only_daily_only"
DEFAULT_DAILY_ONLY_SUMMARY_V2_PRICE_DELTA_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2_price_delta"
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux"
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low"
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_HARD_ST_RUN_TAG = (
    "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low_hard_st"
)
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_HARD_ST_GLOBAL_TAIL_RUN_TAG = (
    "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low_hard_st_global_tail_512"
)
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_PRICE_DELTA_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low_price_delta"
DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_PATH_EQUAL_RUN_TAG = "seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_path_equal"
DEFAULT_RUN_TAG = DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_RUN_TAG
DEFAULT_NO_INTRADAY_RUN_TAG = "seq100_todayclose_path_only_no_intraday_summary"
DEFAULT_NO_LIMIT_RUN_TAG = "seq100_todayclose_path_only_no_limit_structure"
DEFAULT_DIRECT_VALUE_5D_RUN_TAG = "seq100_direct_value_5d"
DEFAULT_DIRECT_VALUE_10D_RUN_TAG = "seq100_direct_value_10d"
DEFAULT_DIRECT_VALUE_60D_RUN_TAG = "seq100_direct_value_60d"
DEFAULT_TOP_K = "1,3,5,10,20,50,100"

ACTIVE_CONCEPTS = {
    "research_store_view": "Lightweight manifest under daily_research/data/research_store/views.",
    "seq100_x32_daily_input": "Past 100 trading days of daily raw and daily state features.",
    "today_close_anchor": "Future path returns are anchored on the signal-day close.",
    "future60_ohlc_path": "The default model predicts the future 60-day OHLC return path.",
    "low_weight_va_auxiliary": "The default model also applies low-weight volume/amount auxiliary path supervision while keeping value/rank price-only.",
    "path_trade_value_v2": "Ranking value is derived from the predicted path, including return, wait, drawdown, and cost semantics.",
    "path_value_spread": "TopK evaluation reports selected path value versus the same-day universe mean; it is not live PnL.",
}

COMPARISON_CONCEPTS = {
    "table_path60_baseline": "Tabular LightGBM-style baseline for aligned path60 comparison.",
    "path_only_next_open": "Retained anchor comparison; not the default mainline.",
    "rank_heavy_top1": "Observation branch for narrow Top1 behavior; not the default ranking objective.",
    "summary_v2_multi_horizon_ohlc": "Explicit experiment that keeps OHLC output but expands summary loss to OHLC-derived 5/10/20/40/60-day constraints.",
    "summary_v2_no60": "Explicit single-factor experiment that keeps summary_v2 OHLC constraints but removes the full 60-day window.",
    "summary_v2_price_delta": "All-channel summary_v2 combination that adds close-to-close log-delta supervision.",
    "summary_v2_ohlcva_aux": "All-channel summary_v2 combination that adds VA auxiliary path supervision and keeps price-only value/rank.",
    "summary_v2_ohlcva_aux_low": "All-channel summary_v2 combination with lower-weight VA auxiliary supervision.",
    "summary_v2_ohlcva_aux_low_price_delta": "All-channel summary_v2 combination with lower-weight VA auxiliary supervision plus close-delta loss.",
    "summary_v2_ohlcva_path_equal": "All-channel summary_v2 combination where OHLCVA fields enter path_loss equally while value/rank remain price-only.",
    "daily_only_no_minute": "Explicit input ablation that keeps labels/loss fixed but removes intraday_summary and limit_structure input channels.",
    "all_channels_base_summary": "Legacy broad-TopK baseline that uses all input channels with the base 60-day summary loss.",
    "daily_only_summary_v2": "Previous default research profile that uses daily-only inputs with multi-horizon OHLC summary_v2 constraints.",
    "daily_only_summary_v2_price_delta": "Explicit price rhythm experiment that adds close-to-close log-delta supervision while keeping OHLC path value semantics.",
    "daily_only_summary_v2_ohlcva_aux": "Explicit volume/amount auxiliary experiment: predicts OHLCVA jointly but keeps price-only summary/value/rank semantics.",
    "daily_only_summary_v2_ohlcva_aux_low_price_delta": "Combined lower-weight VA auxiliary plus close-delta price rhythm experiment.",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st": "Single-factor path-value experiment with hard-max forward semantics and smooth straight-through gradients.",
    "daily_only_summary_v2_ohlcva_aux_low_hard_st_global_tail": "Separated path reconstruction and global-tail daily ranking experiment.",
    "daily_only_summary_v2_ohlcva_path_equal": "Full OHLCVA path reconstruction experiment: six fields enter path_loss equally while value/rank remain price-only.",
    "no_intraday_summary": "Explicit input ablation that removes intraday_summary while retaining limit_structure.",
    "no_limit_structure": "Explicit input ablation that removes limit_structure while retaining intraday_summary.",
    "direct_value_rank_5d": "Explicit sequence ranker that directly learns 5-day path_trade_value_v2 instead of predicting future OHLC.",
    "direct_value_rank_10d": "Explicit sequence ranker that directly learns 10-day path_trade_value_v2 instead of predicting future OHLC.",
    "direct_value_rank_60d": "Explicit sequence ranker that directly learns 60-day path_trade_value_v2 instead of predicting future OHLC.",
}

ARCHIVED_CONCEPTS = {
    "alpha_v2": "Historical infrastructure and evidence line.",
    "path20": "Historical or compatibility horizon; not the current target definition.",
    "symbol_embedding": "Rejected as default until evidence changes.",
    "residual_score": "Comparison branch only.",
    "richer_target": "Comparison branch only.",
    "ohlcva_unified": "Paused branch; higher output complexity did not improve the mainline.",
}


@dataclass(frozen=True)
class TodayClosePathOnlyProfile:
    store_view: Path = DEFAULT_STORE_VIEW
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_tag: str = DEFAULT_RUN_TAG
    epochs: int = 10
    batch_size: int = 512
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    path_loss_weight: float = 0.45
    path_loss_profile: str = PATH_LOSS_PROFILE_DEFAULT
    summary_loss_weight: float = 0.20
    richer_loss_weight: float = 0.0
    price_delta_loss_weight: float = 0.0
    va_level_loss_weight: float = 0.02
    va_delta_loss_weight: float = 0.01
    geometry_loss_weight: float = 0.0
    utility_curve_loss_weight: float = 0.0
    turnover_level_loss_weight: float = 0.0
    turnover_delta_loss_weight: float = 0.0
    value_loss_weight: float = 0.20
    rank_loss_weight: float = 0.15
    rank_max_per_side: int = 64
    device: str = "auto"
    seed: int = DEFAULT_SEED
    prediction_mode: str = "compact"
    evaluation_mode: str = EVALUATION_MODE_STANDARD
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = ""
    early_stopping_mode: str = ""
    min_complete_epochs: int = 1
    development_fixed_final_epoch: bool = False
    top_k: str = DEFAULT_TOP_K
    max_samples_per_split: int = 0
    summary_loss_profile: str = SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC
    input_channel_profile: str = INPUT_CHANNEL_PROFILE_DAILY_ONLY
    model_type: str = "gru_ohlcva_aux_path_value"
    direct_value_horizon: int = 0
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH
    path_value_semantic: str = PATH_VALUE_DEFAULT_SEMANTIC
    rank_training_profile: str = RANK_TRAINING_PROFILE_LOCAL_CHUNK
    rank_batch_size: int = 512
    rank_interval: int = 4
    prefetch_batches: int = 1
    activation_checkpoint_profile: str = "none"


def build_todayclose_path_only_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    argv = [
        "train",
        "--store-view",
        str(profile.store_view),
        "--output-root",
        str(profile.output_root),
        "--run-tag",
        profile.run_tag,
        "--epochs",
        str(profile.epochs),
        "--batch-size",
        str(profile.batch_size),
        "--model-type",
        profile.model_type,
        "--hidden-dim",
        str(profile.hidden_dim),
        "--layers",
        str(profile.layers),
        "--dropout",
        str(profile.dropout),
        "--learning-rate",
        str(profile.learning_rate),
        "--weight-decay",
        str(profile.weight_decay),
        "--path-loss-weight",
        str(profile.path_loss_weight),
        "--path-loss-profile",
        profile.path_loss_profile,
        "--summary-loss-weight",
        str(profile.summary_loss_weight),
        "--richer-loss-weight",
        str(profile.richer_loss_weight),
        "--price-delta-loss-weight",
        str(profile.price_delta_loss_weight),
        "--va-level-loss-weight",
        str(profile.va_level_loss_weight),
        "--va-delta-loss-weight",
        str(profile.va_delta_loss_weight),
        "--geometry-loss-weight",
        str(profile.geometry_loss_weight),
        "--utility-curve-loss-weight",
        str(profile.utility_curve_loss_weight),
        "--turnover-level-loss-weight",
        str(profile.turnover_level_loss_weight),
        "--turnover-delta-loss-weight",
        str(profile.turnover_delta_loss_weight),
        "--value-loss-weight",
        str(profile.value_loss_weight),
        "--rank-loss-weight",
        str(profile.rank_loss_weight),
        "--summary-loss-profile",
        profile.summary_loss_profile,
        "--input-channel-profile",
        profile.input_channel_profile,
        "--direct-value-horizon",
        str(profile.direct_value_horizon),
        "--rank-max-per-side",
        str(profile.rank_max_per_side),
        "--path-value-gradient-profile",
        profile.path_value_gradient_profile,
        "--path-value-semantic",
        profile.path_value_semantic,
        "--rank-training-profile",
        profile.rank_training_profile,
        "--rank-batch-size",
        str(profile.rank_batch_size),
        "--rank-interval",
        str(profile.rank_interval),
        "--prefetch-batches",
        str(profile.prefetch_batches),
        "--activation-checkpoint-profile",
        profile.activation_checkpoint_profile,
        "--device",
        profile.device,
        "--seed",
        str(profile.seed),
        "--top-k",
        profile.top_k,
        "--max-samples-per-split",
        str(profile.max_samples_per_split),
        "--prediction-mode",
        profile.prediction_mode,
        "--evaluation-mode",
        profile.evaluation_mode,
        "--early-stopping-patience",
        str(profile.early_stopping_patience),
        "--early-stopping-min-delta",
        str(profile.early_stopping_min_delta),
        "--min-complete-epochs",
        str(profile.min_complete_epochs),
    ]
    if profile.early_stopping_metric:
        argv.extend(["--early-stopping-metric", profile.early_stopping_metric])
    if profile.early_stopping_mode:
        argv.extend(["--early-stopping-mode", profile.early_stopping_mode])
    if profile.development_fixed_final_epoch:
        argv.append("--development-fixed-final-epoch")
    return argv


@dataclass(frozen=True)
class ProfileSpec:
    command: str
    run_tag: str
    help: str
    input_channel_profile: str
    summary_loss_profile: str
    model_type: str = "gru_path_value"
    path_loss_profile: str = PATH_LOSS_PROFILE_DEFAULT
    path_loss_weight: float = 0.45
    summary_loss_weight: float = 0.20
    richer_loss_weight: float = 0.0
    price_delta_loss_weight: float = 0.0
    va_level_loss_weight: float = 0.0
    va_delta_loss_weight: float = 0.0
    value_loss_weight: float = 0.20
    rank_loss_weight: float = 0.15
    direct_value_horizon: int = 0
    batch_size: int = 512
    early_stopping_patience: int = 2
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH
    rank_training_profile: str = RANK_TRAINING_PROFILE_LOCAL_CHUNK

    def apply(self, profile: TodayClosePathOnlyProfile) -> TodayClosePathOnlyProfile:
        return replace(
            profile,
            path_loss_profile=self.path_loss_profile,
            path_loss_weight=self.path_loss_weight,
            summary_loss_weight=self.summary_loss_weight,
            richer_loss_weight=self.richer_loss_weight,
            price_delta_loss_weight=self.price_delta_loss_weight,
            va_level_loss_weight=self.va_level_loss_weight,
            va_delta_loss_weight=self.va_delta_loss_weight,
            value_loss_weight=self.value_loss_weight,
            rank_loss_weight=self.rank_loss_weight,
            summary_loss_profile=self.summary_loss_profile,
            input_channel_profile=self.input_channel_profile,
            model_type=self.model_type,
            direct_value_horizon=self.direct_value_horizon,
            path_value_gradient_profile=self.path_value_gradient_profile,
            rank_training_profile=self.rank_training_profile,
        )

    def default_profile(self) -> TodayClosePathOnlyProfile:
        return self.apply(
            TodayClosePathOnlyProfile(
                run_tag=self.run_tag,
                batch_size=self.batch_size,
                early_stopping_patience=self.early_stopping_patience,
            )
        )


def _profile_spec(
    command: str,
    run_tag: str,
    help_text: str,
    *,
    input_profile: str,
    summary_profile: str,
    model_type: str = "gru_path_value",
    path_loss_profile: str = PATH_LOSS_PROFILE_DEFAULT,
    price_delta: float = 0.0,
    va_level: float = 0.0,
    va_delta: float = 0.0,
    direct_horizon: int = 0,
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    rank_training_profile: str = RANK_TRAINING_PROFILE_LOCAL_CHUNK,
) -> ProfileSpec:
    direct = int(direct_horizon) > 0
    return ProfileSpec(
        command=command,
        run_tag=run_tag,
        help=help_text,
        input_channel_profile=input_profile,
        summary_loss_profile=summary_profile,
        model_type=model_type,
        path_loss_profile=path_loss_profile,
        path_loss_weight=0.0 if direct else 0.45,
        summary_loss_weight=0.0 if direct else 0.20,
        price_delta_loss_weight=price_delta,
        va_level_loss_weight=va_level,
        va_delta_loss_weight=va_delta,
        value_loss_weight=0.50 if direct else 0.20,
        rank_loss_weight=0.50 if direct else 0.15,
        direct_value_horizon=int(direct_horizon),
        path_value_gradient_profile=path_value_gradient_profile,
        rank_training_profile=rank_training_profile,
        batch_size=2048 if direct else 512,
    )


PROFILE_SPECS = {
    spec.command: spec
    for spec in (
        _profile_spec(
            "train",
            DEFAULT_RUN_TAG,
            "Train the default daily-only summary_v2 today-close path-only mainline with low-weight VA auxiliary supervision.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.02,
            va_delta=0.01,
        ),
        _profile_spec(
            "train-legacy-all-channels-base",
            LEGACY_ALL_CHANNELS_BASE_RUN_TAG,
            "Train the legacy all-channel base-summary broad-TopK baseline.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
        ),
        _profile_spec(
            "train-summary-v2",
            DEFAULT_SUMMARY_V2_RUN_TAG,
            "Train the explicit multi-horizon OHLC summary-loss experiment.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        ),
        _profile_spec(
            "train-summary-v2-no60",
            DEFAULT_SUMMARY_V2_NO60_RUN_TAG,
            "Train the summary_v2 single-factor experiment without the full 60-day window.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
        ),
        _profile_spec(
            "train-summary-v2-price-delta",
            DEFAULT_SUMMARY_V2_PRICE_DELTA_RUN_TAG,
            "Train all-channel summary_v2 with close log-delta price rhythm supervision.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            price_delta=0.03,
        ),
        _profile_spec(
            "train-summary-v2-ohlcva-aux",
            DEFAULT_SUMMARY_V2_OHLCVA_AUX_RUN_TAG,
            "Train all-channel summary_v2 with VA auxiliary path supervision and price-only value/rank.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.05,
            va_delta=0.02,
        ),
        _profile_spec(
            "train-summary-v2-ohlcva-aux-low",
            DEFAULT_SUMMARY_V2_OHLCVA_AUX_LOW_RUN_TAG,
            "Train all-channel summary_v2 with lower-weight VA auxiliary supervision.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.02,
            va_delta=0.01,
        ),
        _profile_spec(
            "train-summary-v2-ohlcva-aux-low-price-delta",
            DEFAULT_SUMMARY_V2_OHLCVA_AUX_LOW_PRICE_DELTA_RUN_TAG,
            "Train all-channel summary_v2 with lower-weight VA auxiliary supervision plus close delta loss.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            price_delta=0.03,
            va_level=0.02,
            va_delta=0.01,
        ),
        _profile_spec(
            "train-summary-v2-ohlcva-path-equal",
            DEFAULT_SUMMARY_V2_OHLCVA_PATH_EQUAL_RUN_TAG,
            "Train all-channel summary_v2 with equal-field OHLCVA path reconstruction and price-only value/rank.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            path_loss_profile=PATH_LOSS_PROFILE_OHLCVA_EQUAL,
        ),
        _profile_spec(
            "train-daily-only",
            DEFAULT_DAILY_ONLY_RUN_TAG,
            "Train the explicit no-minute daily-only input ablation.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
        ),
        _profile_spec(
            "train-daily-only-summary-v2",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_RUN_TAG,
            "Train the daily-only input ablation with multi-horizon OHLC summary_v2 constraints.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-price-delta",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_PRICE_DELTA_RUN_TAG,
            "Train daily-only summary_v2 with close log-delta price rhythm supervision.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            price_delta=0.03,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-aux",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_RUN_TAG,
            "Train the daily-only summary_v2 model with VA auxiliary path supervision and price-only value/rank.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.05,
            va_delta=0.02,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-aux-low",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_RUN_TAG,
            "Train daily-only summary_v2 with lower-weight VA auxiliary supervision.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.02,
            va_delta=0.01,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-aux-low-hard-st",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_HARD_ST_RUN_TAG,
            "Train the daily-only low-VA summary_v2 model with hard-max forward path value and straight-through smooth gradients.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.02,
            va_delta=0.01,
            path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_HARD_ST_GLOBAL_TAIL_RUN_TAG,
            "Train the hard-ST daily-only profile with separated global-tail 512 ranking slates.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            va_level=0.02,
            va_delta=0.01,
            path_value_gradient_profile=PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
            rank_training_profile=RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-aux-low-price-delta",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_AUX_LOW_PRICE_DELTA_RUN_TAG,
            "Train daily-only summary_v2 with lower-weight VA auxiliary supervision plus close delta loss.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            price_delta=0.03,
            va_level=0.02,
            va_delta=0.01,
        ),
        _profile_spec(
            "train-daily-only-summary-v2-ohlcva-path-equal",
            DEFAULT_DAILY_ONLY_SUMMARY_V2_OHLCVA_PATH_EQUAL_RUN_TAG,
            "Train daily-only summary_v2 with equal-field OHLCVA path reconstruction and price-only value/rank.",
            input_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            summary_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            model_type="gru_ohlcva_aux_path_value",
            path_loss_profile=PATH_LOSS_PROFILE_OHLCVA_EQUAL,
        ),
        _profile_spec(
            "train-no-intraday-summary",
            DEFAULT_NO_INTRADAY_RUN_TAG,
            "Train the input ablation that removes intraday_summary only.",
            input_profile=INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
        ),
        _profile_spec(
            "train-no-limit-structure",
            DEFAULT_NO_LIMIT_RUN_TAG,
            "Train the input ablation that removes limit_structure only.",
            input_profile=INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
        ),
        _profile_spec(
            "train-direct-value-5d",
            DEFAULT_DIRECT_VALUE_5D_RUN_TAG,
            "Train the direct 5-day path-value ranker.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
            model_type="gru_direct_value",
            direct_horizon=5,
        ),
        _profile_spec(
            "train-direct-value-10d",
            DEFAULT_DIRECT_VALUE_10D_RUN_TAG,
            "Train the direct 10-day path-value ranker.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
            model_type="gru_direct_value",
            direct_horizon=10,
        ),
        _profile_spec(
            "train-direct-value-60d",
            DEFAULT_DIRECT_VALUE_60D_RUN_TAG,
            "Train the direct 60-day path-value ranker.",
            input_profile=INPUT_CHANNEL_PROFILE_ALL,
            summary_profile=SUMMARY_LOSS_PROFILE_BASE,
            model_type="gru_direct_value",
            direct_horizon=60,
        ),
    )
}


def _profile_argv(
    profile: TodayClosePathOnlyProfile,
    command: str,
    *,
    inherit_price_delta: bool = False,
) -> list[str]:
    configured = PROFILE_SPECS[command].apply(profile)
    if inherit_price_delta:
        configured = replace(configured, price_delta_loss_weight=profile.price_delta_loss_weight)
    return build_todayclose_path_only_train_argv(configured)


def build_todayclose_legacy_all_channels_base_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-legacy-all-channels-base")


def build_todayclose_summary_v2_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2")


def build_todayclose_summary_v2_price_delta_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-price-delta", inherit_price_delta=True)


def build_todayclose_summary_v2_ohlcva_aux_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-ohlcva-aux")


def build_todayclose_summary_v2_ohlcva_aux_low_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-ohlcva-aux-low")


def build_todayclose_summary_v2_ohlcva_aux_low_price_delta_train_argv(
    profile: TodayClosePathOnlyProfile,
) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-ohlcva-aux-low-price-delta", inherit_price_delta=True)


def build_todayclose_summary_v2_ohlcva_path_equal_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-ohlcva-path-equal")


def build_todayclose_summary_v2_no60_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-summary-v2-no60")


def build_todayclose_no_intraday_summary_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-no-intraday-summary")


def build_todayclose_no_limit_structure_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-no-limit-structure")


def build_todayclose_daily_only_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-daily-only")


def build_todayclose_daily_only_summary_v2_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2")


def build_todayclose_daily_only_summary_v2_price_delta_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-price-delta", inherit_price_delta=True)


def build_todayclose_daily_only_summary_v2_ohlcva_aux_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-ohlcva-aux", inherit_price_delta=True)


def build_todayclose_daily_only_summary_v2_ohlcva_aux_low_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-ohlcva-aux-low")


def build_todayclose_daily_only_summary_v2_ohlcva_aux_low_hard_st_train_argv(
    profile: TodayClosePathOnlyProfile,
) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-ohlcva-aux-low-hard-st")


def build_todayclose_daily_only_summary_v2_ohlcva_aux_low_hard_st_global_tail_train_argv(
    profile: TodayClosePathOnlyProfile,
) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail")


def build_todayclose_daily_only_summary_v2_ohlcva_aux_low_price_delta_train_argv(
    profile: TodayClosePathOnlyProfile,
) -> list[str]:
    return _profile_argv(
        profile,
        "train-daily-only-summary-v2-ohlcva-aux-low-price-delta",
        inherit_price_delta=True,
    )


def build_todayclose_daily_only_summary_v2_ohlcva_path_equal_train_argv(
    profile: TodayClosePathOnlyProfile,
) -> list[str]:
    return _profile_argv(profile, "train-daily-only-summary-v2-ohlcva-path-equal")


def build_todayclose_direct_value_train_argv(profile: TodayClosePathOnlyProfile, *, horizon: int) -> list[str]:
    configured = replace(
        profile,
        path_loss_weight=0.0,
        path_loss_profile=PATH_LOSS_PROFILE_DEFAULT,
        summary_loss_weight=0.0,
        richer_loss_weight=0.0,
        price_delta_loss_weight=0.0,
        va_level_loss_weight=0.0,
        va_delta_loss_weight=0.0,
        value_loss_weight=0.50,
        rank_loss_weight=0.50,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_BASE,
        input_channel_profile=INPUT_CHANNEL_PROFILE_ALL,
        model_type="gru_direct_value",
        direct_value_horizon=int(horizon),
    )
    return build_todayclose_path_only_train_argv(configured)


def _contract_profile(command: str) -> dict[str, Any]:
    return asdict(PROFILE_SPECS[command].default_profile())


def mainline_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mainline_id": "seq100_todayclose_path_only",
        "default_store_view": str(DEFAULT_STORE_VIEW),
        "active_concepts": ACTIVE_CONCEPTS,
        "comparison_concepts": COMPARISON_CONCEPTS,
        "archived_concepts": ARCHIVED_CONCEPTS,
        "default_train_profile": _contract_profile("train"),
        "legacy_all_channels_base_train_profile": _contract_profile("train-legacy-all-channels-base"),
        "summary_v2_train_profile": _contract_profile("train-summary-v2"),
        "summary_v2_no60_train_profile": _contract_profile("train-summary-v2-no60"),
        "summary_v2_price_delta_train_profile": _contract_profile("train-summary-v2-price-delta"),
        "summary_v2_ohlcva_aux_train_profile": _contract_profile("train-summary-v2-ohlcva-aux"),
        "summary_v2_ohlcva_aux_low_train_profile": _contract_profile("train-summary-v2-ohlcva-aux-low"),
        "summary_v2_ohlcva_aux_low_price_delta_train_profile": _contract_profile(
            "train-summary-v2-ohlcva-aux-low-price-delta"
        ),
        "summary_v2_ohlcva_path_equal_train_profile": _contract_profile("train-summary-v2-ohlcva-path-equal"),
        "daily_only_train_profile": _contract_profile("train-daily-only"),
        "daily_only_summary_v2_train_profile": _contract_profile("train-daily-only-summary-v2"),
        "daily_only_summary_v2_price_delta_train_profile": _contract_profile(
            "train-daily-only-summary-v2-price-delta"
        ),
        "daily_only_summary_v2_ohlcva_aux_train_profile": _contract_profile(
            "train-daily-only-summary-v2-ohlcva-aux"
        ),
        "daily_only_summary_v2_ohlcva_aux_low_train_profile": _contract_profile(
            "train-daily-only-summary-v2-ohlcva-aux-low"
        ),
        "daily_only_summary_v2_ohlcva_aux_low_price_delta_train_profile": _contract_profile(
            "train-daily-only-summary-v2-ohlcva-aux-low-price-delta"
        ),
        "daily_only_summary_v2_ohlcva_path_equal_train_profile": _contract_profile(
            "train-daily-only-summary-v2-ohlcva-path-equal"
        ),
        "input_ablation_train_profiles": {
            "no_intraday_summary": _contract_profile("train-no-intraday-summary"),
            "no_limit_structure": _contract_profile("train-no-limit-structure"),
        },
        "direct_value_train_profiles": {
            "5d": _contract_profile("train-direct-value-5d"),
            "10d": _contract_profile("train-direct-value-10d"),
            "60d": _contract_profile("train-direct-value-60d"),
        },
        "evidence_boundary": (
            "Sequence path-value metrics are research evidence. They do not activate "
            "active_execution_strategy.json, live/default, broker, or trade-plan changes."
        ),
        "primary_evaluation_policy": {
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "train_label_rule": "dependency_end_trade_date < development_start_trade_date",
            "checkpoint_policy": "best_development_total_loss",
            "development_access": "evaluate_after_each_complete_epoch_for_early_stopping_and_model_selection",
            "development_years": [2022, 2023, 2024, 2025],
            "seed": 7,
            "training_sample_policy": "all_rows",
            "minimum_complete_epochs": 1,
        },
        "legacy_fixed_oos_policy": {
            "method": "purged_expanding_walk_forward",
            "split_roles": {"fit": "train", "evaluation": "oos"},
            "checkpoint_policy": "final_epoch",
            "status": "compatibility_only",
        },
        "next_decision_surface": "full_candidate_development_walkforward_or_loss_redesign",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_sequence_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing sequence_path_training_summary.json under {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_rows = _read_csv_rows(run_dir / "split_metrics.csv")
    topk_rows = _read_csv_rows(run_dir / "topk_metrics.csv")
    value_column = str(summary.get("value_column", "") or "")

    slim_splits = {}
    for row in split_rows:
        split = row.get("split", "")
        if split:
            slim_splits[split] = {
                "row_count": int(float(row.get("row_count") or 0)),
                "date_count": int(float(row.get("date_count") or 0)),
                "rank_ic_mean": float(row.get("rank_ic_mean") or "nan"),
                "rank_ic_positive_day_rate": float(row.get("rank_ic_positive_day_rate") or "nan"),
                "value_column": row.get("value_column", summary.get("value_column", "")),
            }

    slim_topk: dict[str, dict[str, dict[str, float]]] = {split: {} for split in slim_splits}
    for row in topk_rows:
        split = row.get("split", "")
        top_k = row.get("top_k", "")
        if split in slim_topk and top_k in {"1", "3", "10"}:
            slim_topk[split][top_k] = {
                "path_value_spread": float(
                    row.get(f"alpha_{value_column}")
                    or row.get("alpha_path_trade_value_v2_60d")
                    or row.get("alpha_path_trade_value_60d")
                    or "nan"
                ),
                "best_exit_day_mean": float(row.get("selected_best_exit_day_mean") or "nan"),
                "hit_10pct_rate": float(row.get("selected_hit_10pct_rate") or "nan"),
                "loss_5pct_rate": float(row.get("selected_loss_5pct_rate") or "nan"),
            }

    return {
        "run_dir": str(run_dir),
        "generated_at": summary.get("generated_at", ""),
        "run_tag": summary.get("run_tag", ""),
        "seed": summary.get("seed", ""),
        "fold_year": summary.get("fold_year", ""),
        "evaluation_mode": summary.get("evaluation_mode", ""),
        "checkpoint_policy": summary.get("checkpoint_policy", ""),
        "pack_manifest": summary.get("pack_manifest", ""),
        "input_channel_profile": summary.get("input_channel_profile", ""),
        "input_channels": summary.get("input_channels", []),
        "model_input_dim": dict(summary.get("model", {}) or {}).get("input_dim", ""),
        "price_anchor": summary.get("price_anchor", ""),
        "value_column": summary.get("value_column", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "loss_weights": summary.get("loss_weights", {}),
        "splits": slim_splits,
        "topk": slim_topk,
        "evidence_boundary": mainline_contract()["evidence_boundary"],
    }


def _print_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrow CLI for the seq100 today-close path-only mainline.")
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="Print the active concept contract.")
    contract.add_argument("--json", action="store_true")

    for spec in PROFILE_SPECS.values():
        train = sub.add_parser(spec.command, help=spec.help)
        _add_train_args(
            train,
            default_run_tag=spec.run_tag,
            default_early_stopping_patience=spec.early_stopping_patience,
            default_batch_size=spec.batch_size,
        )

    summarize = sub.add_parser("summarize", help="Print a compact sequence run summary.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--json", action="store_true")
    return parser


def _add_train_args(
    parser: argparse.ArgumentParser,
    *,
    default_run_tag: str,
    default_early_stopping_patience: int,
    default_batch_size: int = 512,
) -> None:
    parser.add_argument("--store-view", type=Path, default=DEFAULT_STORE_VIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=default_run_tag)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=int(default_batch_size))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Training-only complete-date screening cap; evaluation splits remain full-universe.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=int(default_early_stopping_patience))
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-metric", default="")
    parser.add_argument("--early-stopping-mode", default="")
    parser.add_argument("--min-complete-epochs", type=int, default=1)
    parser.add_argument("--prediction-mode", default="compact", choices=("full", "compact", "none"))
    parser.add_argument("--evaluation-mode", default=EVALUATION_MODE_STANDARD, choices=EVALUATION_MODES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "contract":
        _print_payload(mainline_contract(), as_json=bool(args.json))
        return 0
    if args.command == "summarize":
        _print_payload(summarize_sequence_run(Path(args.run_dir)), as_json=bool(args.json))
        return 0

    spec = PROFILE_SPECS.get(str(args.command))
    if spec is None:
        parser.error(f"unsupported command: {args.command}")
        return 2

    base_profile = TodayClosePathOnlyProfile(
        store_view=Path(args.store_view),
        output_root=Path(args.output_root),
        run_tag=str(args.run_tag),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=str(args.device),
        seed=int(args.seed),
        prediction_mode=str(args.prediction_mode),
        evaluation_mode=str(args.evaluation_mode),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        early_stopping_metric=str(args.early_stopping_metric),
        early_stopping_mode=str(args.early_stopping_mode),
        min_complete_epochs=int(args.min_complete_epochs),
        max_samples_per_split=int(args.max_samples_per_split),
    )
    profile = spec.apply(base_profile)
    train_argv = build_todayclose_path_only_train_argv(profile)
    if bool(args.dry_run):
        _print_payload({"profile": asdict(profile), "argv": train_argv}, as_json=bool(args.json))
        return 0
    if bool(args.json):
        train_argv.append("--json")
    return sequence_training_main(train_argv)


if __name__ == "__main__":
    raise SystemExit(main())
