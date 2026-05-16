import inspect
import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy import pipeline_utils
from daily_research.continuous_policy.decision_core_v6 import (
    DECISION_CORE_V6_ARTIFACT_FILENAME,
    DECISION_CORE_V6_ARTIFACT_TYPE,
    DECISION_CORE_V6_PROFILE,
    DECISION_CORE_V6_STANDARD_COLUMNS,
    TorchDecisionCoreV6Artifact,
    attach_v6_feature_contract,
    build_decision_frame_v6,
    build_decision_core_v6_targets,
    load_decision_core_v6_artifact,
)
from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
    build_training_contract,
    normalize_trainer_backend,
)


class DecisionCoreV6Test(unittest.TestCase):
    def _state(self, include_high_value: bool = True) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "stock": ["SRC", "RCV", "HLD"],
                "score_blend": [0.1, 0.9, 0.5],
                "score_rank_pct": [0.2, 0.95, 0.5],
                "in_pool": [1.0, 1.0, 1.0],
                "current_weight": [0.12, 0.0, 0.03],
                "holding_flag": [1.0, 0.0, 1.0],
                "ret_1d": [-0.01, 0.02, 0.0],
                "ret_3d": [-0.03, 0.04, 0.0],
                "ret_5d": [-0.04, 0.06, 0.0],
                "vol_20d": [0.04, 0.02, 0.03],
                "close": [10.0, 20.0, 30.0],
                "open": [9.9, 19.8, 29.8],
                "high": [10.2, 20.2, 30.2],
                "low": [9.7, 19.6, 29.6],
            }
        )
        if include_high_value:
            frame = frame.assign(
                adv20=[1000.0, 2000.0, 1500.0],
                amount=[1.0e6, 2.0e6, 1.5e6],
                volume=[100.0, 200.0, 150.0],
                z_drawdown_20=[0.8, 0.0, 0.2],
                z_volatility_20=[0.5, 0.1, 0.2],
                z_vol_ratio_5_20=[0.2, 0.0, 0.1],
                z_price_volume_divergence=[0.1, 0.0, 0.1],
                z_breakout_volume=[0.0, 1.0, 0.2],
                z_volatility_contraction=[0.0, 0.2, 0.1],
                z_volume_contraction=[0.0, 0.1, 0.1],
            )
        return frame

    def _artifact(self) -> TorchDecisionCoreV6Artifact:
        return TorchDecisionCoreV6Artifact(
            feature_names=["score_blend", "current_weight"],
            daily_feature_names=["market_downside_pressure"],
            static_feature_names=["score_blend", "current_weight"],
            sequence_bases=[],
            feature_fill_values=np.zeros(2, dtype=np.float32),
            feature_means=np.zeros(2, dtype=np.float32),
            feature_stds=np.ones(2, dtype=np.float32),
            sequence_fill_values=np.zeros(1, dtype=np.float32),
            sequence_means=np.zeros(1, dtype=np.float32),
            sequence_stds=np.ones(1, dtype=np.float32),
            daily_fill_values=np.zeros(1, dtype=np.float32),
            daily_means=np.zeros(1, dtype=np.float32),
            daily_stds=np.ones(1, dtype=np.float32),
            train_summary={"run_tag": "decision_core_v6_test", "trainer_backend": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6},
            training_diagnostics={"artifact_schema": DECISION_CORE_V6_ARTIFACT_TYPE},
            training_contract={"trainer_backend": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6, "promotable": False},
            trained_at="2026-05-15T00:00:00+08:00",
            model_config={
                "model_dim": 16,
                "temporal_layers": 1,
                "cross_layers": 1,
                "latent_count": 4,
                "dropout": 0.0,
                "decision_core_profile": DECISION_CORE_V6_PROFILE,
            },
            global_target_defaults={"gross_exposure_target": 0.7, "turnover_budget": 0.08},
        )

    def test_training_contract_is_shadow_only_longrun_v6(self) -> None:
        self.assertEqual(normalize_trainer_backend("decision_core_v6"), TRAINER_BACKEND_FORMAL_DECISION_CORE_V6)

        contract = build_training_contract(
            trainer_backend=TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
            requested_epochs=1,
            min_epochs=1,
        )

        self.assertEqual(contract["trainer_backend"], TRAINER_BACKEND_FORMAL_DECISION_CORE_V6)
        self.assertFalse(contract["promotable"])
        self.assertTrue(contract["resume_capable"])
        self.assertGreaterEqual(contract["requested_epochs"], 32)
        self.assertGreaterEqual(contract["min_epochs"], 32)

    def test_feature_contract_blocker_neutral_and_ok(self) -> None:
        ok = attach_v6_feature_contract(self._state(include_high_value=True))
        self.assertTrue((ok["decision_core_v6_feature_contract_status"] == "ok").all())
        self.assertEqual(float(ok["decision_core_v6_feature_contract_blocker_count"].max()), 0.0)

        neutral = attach_v6_feature_contract(self._state(include_high_value=False))
        self.assertEqual(float(neutral["decision_core_v6_feature_contract_blocker_count"].max()), 0.0)
        self.assertEqual(float(neutral["decision_core_v6_feature_contract_degraded_count"].max()), 0.0)
        self.assertGreater(float(neutral["decision_core_v6_feature_contract_neutral_fallback_count"].max()), 0.0)

        blocker = attach_v6_feature_contract(self._state().drop(columns=["score_blend", "score_rank_pct"]))
        self.assertTrue((blocker["decision_core_v6_feature_contract_status"] == "blocker").all())
        self.assertGreater(float(blocker["decision_core_v6_feature_contract_blocker_count"].max()), 0.0)

    def test_feature_contract_uses_feature_family_health_not_raw_ohlc_only(self) -> None:
        derived_state = self._state(include_high_value=False).drop(columns=["close", "open", "high", "low"])
        derived_state = derived_state.assign(
            current_price=[10.0, 20.0, 30.0],
            price_rank=[0.2, 0.8, 0.5],
        )

        contract = attach_v6_feature_contract(derived_state)

        self.assertTrue((contract["decision_core_v6_feature_contract_status"] == "ok").all())
        self.assertEqual(float(contract["decision_core_v6_feature_contract_degraded_count"].max()), 0.0)
        self.assertGreater(float(contract["decision_core_v6_feature_contract_neutral_fallback_count"].max()), 0.0)

        degraded = attach_v6_feature_contract(
            derived_state.drop(
                columns=[
                    "current_price",
                    "price_rank",
                    "ret_1d",
                    "ret_3d",
                    "ret_5d",
                    "vol_20d",
                ]
            )
        )
        self.assertTrue((degraded["decision_core_v6_feature_contract_status"] == "degraded").all())
        self.assertGreater(float(degraded["decision_core_v6_feature_contract_degraded_count"].max()), 0.0)

    def test_decision_frame_v6_outputs_standard_columns_and_cashflow_valid(self) -> None:
        state = self._state()
        outputs = pd.DataFrame(
            {
                "source_supply_score": [0.95, 0.05, 0.10],
                "receiver_demand_score": [0.05, 0.95, 0.10],
                "cash_buffer_score": [0.05, 0.05, 0.05],
                "target_weight": [0.03, 0.08, 0.03],
                "target_delta": [-0.09, 0.08, 0.0],
                "release_intent": [0.9, 0.0, 0.0],
                "reduce_quality": [0.9, 0.0, 0.0],
                "exit_hazard": [0.1, 0.0, 0.0],
            },
            index=state.index,
        )

        decision = build_decision_frame_v6(state_frame=state, outputs=outputs, turnover_budget=0.08)

        for column in DECISION_CORE_V6_STANDARD_COLUMNS:
            self.assertIn(column, decision.frame.columns)
        self.assertEqual(str(decision.frame["decision_core_version"].iloc[0]), "v6")
        self.assertEqual(float(decision.frame["portfolio_cashflow_decision_v1_valid"].iloc[0]), 1.0)
        self.assertLessEqual(float(decision.frame["decision_core_v6_oracle_constraint_violation"].iloc[0]), 1.0e-6)
        self.assertGreater(float(decision.frame["portfolio_set_v5_source_supply"].sum()), 0.0)
        self.assertGreater(float(decision.frame["portfolio_set_v5_receiver_demand"].sum()), 0.0)

    def test_decision_frame_v6_protects_strong_keep_and_releases_weak_sources(self) -> None:
        state = pd.DataFrame(
            {
                "stock": ["WEAK", "STRONG", "RCV"],
                "score_blend": [0.05, 0.95, 0.90],
                "score_rank_pct": [0.10, 0.98, 0.92],
                "in_pool": [1.0, 1.0, 1.0],
                "current_weight": [0.12, 0.12, 0.0],
                "holding_flag": [1.0, 1.0, 0.0],
                "ret_1d": [-0.01, 0.02, 0.02],
                "ret_3d": [-0.03, 0.05, 0.05],
                "ret_5d": [-0.08, 0.10, 0.08],
                "ret_20d": [-0.16, 0.18, 0.16],
                "score_delta_1d": [-0.20, 0.25, 0.20],
                "score_delta_5d": [-0.60, 0.70, 0.55],
                "vol_20d": [0.05, 0.03, 0.03],
                "distance_to_20d_high": [-0.20, -0.01, -0.02],
                "drawdown_from_peak": [-0.18, -0.01, 0.0],
                "unrealized_pnl": [-0.10, 0.20, 0.0],
                "recent_buy_flag": [0.0, 1.0, 0.0],
                "holding_age_short": [0.0, 1.0, 0.0],
                "hold_continuity_pressure": [0.0, 0.90, 0.0],
                "close": [10.0, 20.0, 30.0],
                "open": [9.9, 19.8, 29.8],
                "high": [10.2, 20.2, 30.2],
                "low": [9.7, 19.6, 29.6],
            }
        )
        outputs = pd.DataFrame(
            {
                "source_supply_score": [0.95, 0.95, 0.0],
                "receiver_demand_score": [0.0, 0.0, 0.95],
                "cash_buffer_score": [0.05, 0.05, 0.05],
                "target_weight": [0.03, 0.03, 0.08],
                "target_delta": [-0.09, -0.09, 0.08],
                "release_intent": [0.9, 0.9, 0.0],
                "reduce_quality": [0.9, 0.9, 0.0],
                "exit_hazard": [0.0, 0.0, 0.0],
            },
            index=state.index,
        )

        decision = build_decision_frame_v6(state_frame=state, outputs=outputs, turnover_budget=0.08).frame

        weak_supply = float(decision.loc[0, "source_supply"])
        strong_supply = float(decision.loc[1, "source_supply"])
        self.assertGreater(weak_supply, strong_supply + 0.01)
        self.assertGreater(float(decision.loc[0, "decision_core_v6_source_weakness"]), 0.45)
        self.assertGreater(float(decision.loc[1, "decision_core_v6_source_keep_strength"]), 0.65)
        self.assertGreaterEqual(float(decision.loc[1, "target_delta"]), -0.003)
        self.assertGreater(float(decision.loc[2, "receiver_demand"]), 0.0)

    def test_decision_frame_v6_release_gate_blocks_high_keep_risk_source(self) -> None:
        state = pd.DataFrame(
            {
                "stock": ["KEEP", "WEAK", "RCV"],
                "score_blend": [0.92, 0.05, 0.95],
                "score_rank_pct": [0.95, 0.05, 0.98],
                "in_pool": [1.0, 1.0, 1.0],
                "current_weight": [0.12, 0.12, 0.0],
                "holding_flag": [1.0, 1.0, 0.0],
                "ret_1d": [0.02, -0.02, 0.02],
                "ret_3d": [0.04, -0.04, 0.04],
                "ret_5d": [0.08, -0.08, 0.08],
                "ret_20d": [0.15, -0.15, 0.15],
                "score_delta_5d": [0.45, -0.55, 0.45],
                "vol_20d": [0.02, 0.05, 0.02],
                "portfolio_daily_source_forward_proxy_keep_risk": [0.90, 0.08, 0.0],
                "portfolio_daily_source_positive_forward_penalty": [0.92, 0.05, 0.0],
                "portfolio_daily_source_opportunity_cost_penalty": [0.90, 0.05, 0.0],
                "portfolio_daily_source_forward_strength_brake_risk": [0.88, 0.05, 0.0],
                "portfolio_daily_source_release_conviction": [0.18, 0.72, 0.0],
                "hold_continuity_pressure": [0.90, 0.05, 0.0],
                "close": [10.0, 20.0, 30.0],
                "open": [9.9, 19.8, 29.8],
                "high": [10.2, 20.2, 30.2],
                "low": [9.7, 19.6, 29.6],
            }
        )
        outputs = pd.DataFrame(
            {
                "source_supply_score": [0.98, 0.98, 0.0],
                "receiver_demand_score": [0.0, 0.0, 0.98],
                "cash_buffer_score": [0.05, 0.05, 0.05],
                "target_weight": [0.03, 0.03, 0.08],
                "target_delta": [-0.09, -0.09, 0.08],
                "release_intent": [0.95, 0.95, 0.0],
                "reduce_quality": [0.95, 0.95, 0.0],
                "exit_hazard": [0.0, 0.0, 0.0],
            },
            index=state.index,
        )

        decision = build_decision_frame_v6(state_frame=state, outputs=outputs, turnover_budget=0.08).frame

        self.assertLess(float(decision.loc[0, "decision_core_v6_release_gate"]), 0.5)
        self.assertLess(float(decision.loc[0, "source_supply"]), 0.003)
        self.assertGreater(float(decision.loc[1, "decision_core_v6_release_gate"]), 0.5)
        self.assertGreater(float(decision.loc[1, "source_supply"]), 0.003)

    def test_targets_project_each_day_independently(self) -> None:
        day_a = self._state().assign(date="2020-01-02")
        day_b = self._state().assign(date="2020-01-03")
        two_days = pd.concat([day_a, day_b], ignore_index=True)

        targets = build_decision_core_v6_targets(two_days)

        for _, day in targets.groupby(two_days["date"], sort=True):
            turnover = float(day["target_delta"].abs().sum())
            self.assertLessEqual(turnover, float(day["turnover_budget"].iloc[0]) + 1.0e-6)
            self.assertLessEqual(float(day["constraint_violation"].max()), 1.0e-6)

    def test_targets_include_dataset_adapter_columns(self) -> None:
        targets = build_decision_core_v6_targets(self._state())

        for column in ("held_mask", "source_mask", "receiver_mask", "source_supply", "receiver_demand", "target_weight"):
            self.assertIn(column, targets.columns)
        self.assertEqual(str(targets["decision_core_version"].iloc[0]), "v6")

    def test_artifact_round_trips_through_generic_loader_and_predict(self) -> None:
        artifact = self._artifact()
        from daily_research.continuous_policy.decision_core_v6 import _make_v6_model

        state = {key: value.detach().clone() for key, value in _make_v6_model(artifact).state_dict().items()}
        state["head.weight"] = torch.zeros_like(state["head.weight"])
        state["head.bias"] = torch.tensor([0.0, 0.0, 2.4, 2.4, -3.0, 0.0, 0.0, 0.0], dtype=state["head.bias"].dtype)
        artifact.model_state_dict = state
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / DECISION_CORE_V6_ARTIFACT_FILENAME
            artifact.save(path)
            loaded_direct = load_decision_core_v6_artifact(path)
            loaded_generic = load_artifact(path)

        self.assertIsInstance(loaded_direct, TorchDecisionCoreV6Artifact)
        self.assertIsInstance(loaded_generic, TorchDecisionCoreV6Artifact)
        policy, global_targets = predict_policy(
            loaded_generic,
            state_frame=self._state(),
            daily_features={"market_downside_pressure": 0.1},
        )
        self.assertEqual(global_targets["decision_core_version"], "v6")
        self.assertIn("decision_core_v6_intent", policy.columns)
        self.assertIn("portfolio_cashflow_decision_v1_mode", policy.columns)

    def test_continuity_metrics_use_narrow_held_decision_slice_for_wide_v6_frames(self) -> None:
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
        stocks = ["SRC", "RCV", "HLD"]
        close = pd.DataFrame(
            {
                "SRC": [10.0, 9.8, 9.7, 9.6],
                "RCV": [20.0, 20.4, 20.7, 21.0],
                "HLD": [30.0, 30.2, 30.3, 30.4],
            },
            index=dates,
        )
        prepared = PreparedPolicyInputs(
            universe=tuple(stocks),
            pool_name="unit",
            benchmark="000300.SH",
            data_source="lake",
            csv_folder="",
            start_date="20200101",
            end_date="20200106",
            requested_start_date="20200101",
            history_window=HistoryWindow(
                mode="explicit",
                requested_start_date="20200101",
                effective_start_date="20200101",
                end_date="20200106",
                required_trading_days=4,
            ),
            raw_cache_meta={},
            prepared_cache_meta={},
            close=close,
            open_=close,
            high=close,
            low=close,
            volume=close,
            amount=close,
            benchmark_close=pd.Series([100.0, 99.0, 100.0, 101.0], index=dates),
            benchmark_open=pd.Series([100.0, 99.0, 100.0, 101.0], index=dates),
            score_none=close,
            score_v2=close,
            score_blend=close,
            feature_frames={},
            market_features={},
            membership_frame=pd.DataFrame(1.0, index=dates, columns=stocks),
            rolling_pool_summary={},
            alpha_prior_summary={},
            derived_frames={},
        )
        wide_outcomes = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-02"],
                "stock": ["SRC", "RCV", "HLD", "SRC", "RCV", "HLD"],
                "execution_action": ["exit", "add", "hold", "reduce", "open", "hold"],
                "hold_days_before": [4.0, 2.0, 3.0, 5.0, 0.0, 4.0],
                "forward_excess_5d": [-0.03, 0.04, 0.02, -0.02, 0.03, 0.01],
                "forward_benchmark_return_1d": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
                "future_max_up_10d": [0.01, 0.09, 0.04, 0.02, 0.08, 0.03],
                "future_min_down_10d": [-0.05, -0.01, -0.02, -0.04, -0.01, -0.01],
                "unrealized_pnl_before": [0.08, 0.01, 0.03, 0.06, 0.0, 0.02],
                "drawdown_from_peak_before": [0.02, 0.00, 0.01, 0.03, 0.0, 0.01],
                "model_action": ["exit", "add", "hold", "reduce", "open", "hold"],
                "weight_change_action": ["exit", "add", "hold", "reduce", "open", "hold"],
                "decision_core_version": ["v6"] * 6,
                "decision_core_v6_contract_valid": [1.0] * 6,
                "decision_core_v6_oracle_constraint_violation": [0.0] * 6,
                "decision_core_v6_feature_contract_blocker_count": [0.0] * 6,
                "decision_core_v6_feature_contract_degraded_count": [0.0] * 6,
                "decision_core_v6_feature_contract_neutral_fallback_count": [0.0] * 6,
                "portfolio_cashflow_decision_v1_mode": [1.0] * 6,
                "portfolio_cashflow_decision_v1_valid": [1.0] * 6,
                "portfolio_cashflow_decision_v1_cash_conservation_gap": [0.0] * 6,
                "portfolio_daily_source_target": [True, False, False, True, False, False],
                "portfolio_daily_receiver_target": [False, True, False, False, True, False],
                "portfolio_daily_source_target_intent": [True, False, False, True, False, False],
                "portfolio_daily_receiver_target_intent": [False, True, False, False, True, False],
            }
        )
        wide_outcomes = pd.concat(
            [
                wide_outcomes,
                pd.DataFrame(
                    {f"wide_unused_{column_idx:03d}": [float(column_idx)] * len(wide_outcomes) for column_idx in range(240)}
                ),
            ],
            axis=1,
        )

        with patch(
            "daily_research.continuous_policy.pipeline_utils.build_action_outcome_frame",
            return_value=wide_outcomes,
        ):
            metrics, returned = pipeline_utils.compute_continuity_metrics(
                prepared=prepared,
                action_panel=pd.DataFrame({"unused": []}),
                turnover_frame=pd.DataFrame(),
                returns_series=pd.Series([0.0, 0.01], index=dates[:2]),
            )

        self.assertIn("wide_unused_239", returned.columns)
        self.assertEqual(metrics["decision_core_v6_mode_count"], 6.0)
        self.assertEqual(metrics["decision_core_v6_contract_valid_rate"], 1.0)
        self.assertEqual(metrics["feature_contract_blocker_rate"], 0.0)
        self.assertGreater(metrics["portfolio_daily_receiver_target_count"], 0.0)
        self.assertGreater(metrics["portfolio_daily_source_target_count"], 0.0)

    def test_continuity_metrics_do_not_restore_full_width_held_decision_copy(self) -> None:
        source = inspect.getsource(pipeline_utils.compute_continuity_metrics)

        self.assertIn("_narrow_action_rows", source)
        self.assertNotIn("held_decision_rows = action_outcomes.loc[", source)
        self.assertNotIn("held_rank_rows = held_decision_rows.copy()", source)

    def test_strict_resume_reuses_completed_v6_training_artifact(self) -> None:
        from daily_research.continuous_policy.runtime_progress import JsonlProgressSink
        from daily_research.continuous_policy.train_policy import _try_reuse_decision_core_v6_training

        with TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            artifact_path = run_root / DECISION_CORE_V6_ARTIFACT_FILENAME
            self._artifact().save(artifact_path)
            summary = {
                "run_tag": "decision_core_v6_resume_unit",
                "trainer_backend": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
                "loss_profile": DECISION_CORE_V6_PROFILE,
                "training_contract": {
                    "trainer_backend": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
                    "min_epochs": 32,
                },
                "training_dataset_cache": {
                    "dataset_id": "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1"
                },
                "training_diagnostics": {
                    "artifact_schema": DECISION_CORE_V6_ARTIFACT_TYPE,
                    "completed_epochs": 32,
                    "best_epoch": 27,
                    "training_dataset_id": "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1",
                },
                "model_artifact_path": str(artifact_path),
            }
            (run_root / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch("daily_research.continuous_policy.train_policy.update_latest_summary") as update_latest:
                reused = _try_reuse_decision_core_v6_training(
                    run_root=run_root,
                    args=Namespace(
                        resume_mode="strict",
                        training_dataset_id="continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1",
                    ),
                    training_contract={"min_epochs": 32},
                    progress_sink=JsonlProgressSink(None, run_tag="decision_core_v6_resume_unit", stage="train"),
                )

        self.assertIsNotNone(reused)
        self.assertTrue(reused["training_diagnostics"]["strict_resume_reused_artifact"])
        update_latest.assert_called_once()

    def test_strict_resume_main_stops_prepare_heartbeat_before_return(self) -> None:
        import daily_research.continuous_policy.train_policy as train_module

        class StopAfterSafePrint(Exception):
            pass

        class FakeThread:
            instances: list["FakeThread"] = []

            def __init__(self, *args, **kwargs) -> None:
                self.started = False
                self.joined = False
                FakeThread.instances.append(self)

            def start(self) -> None:
                self.started = True

            def join(self, timeout: float | None = None) -> None:
                self.joined = True

        argv = [
            "--decision-core",
            "v6",
            "--data-source",
            "lake",
            "--training-dataset-id",
            "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1",
            "--lake-dataset-id",
            "policy_input_bundle__fixture",
            "--tag",
            "decision_core_v6_resume_unit",
            "--protocol-progress-jsonl",
            "progress.jsonl",
        ]
        with TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "progress.jsonl"
            argv[argv.index("progress.jsonl")] = str(progress_path)
            with (
                patch.object(train_module, "MODELS_ROOT", Path(temp_dir)),
                patch.object(train_module.threading, "Thread", FakeThread),
                patch.object(train_module, "_try_reuse_decision_core_v6_training", return_value={"status": "reused"}),
                patch.object(train_module, "safe_print_json", side_effect=StopAfterSafePrint),
            ):
                with self.assertRaises(StopAfterSafePrint):
                    train_module.main(argv)

        self.assertEqual(len(FakeThread.instances), 1)
        self.assertTrue(FakeThread.instances[0].started)
        self.assertTrue(FakeThread.instances[0].joined)

    def test_training_evidence_counts_non_skip_teacher_distribution_for_sharded_gold(self) -> None:
        from daily_research.continuous_policy.run_continuous_policy_protocol import _build_training_evidence_assessment

        assessment = _build_training_evidence_assessment(
            {
                "daily_rows": 3949,
                "training_diagnostics": {
                    "completed_epochs": 32,
                    "best_epoch": 27,
                    "train_day_count": 256,
                    "validation_day_count": 31,
                },
                "teacher_summary": {
                    "teacher_action_distribution": {
                        "add": 2800,
                        "exit": 3967,
                        "hold": 152215,
                        "open": 317734,
                        "reduce": 10110,
                        "skip": 1376642,
                    }
                },
            }
        )

        self.assertEqual(assessment["teacher_action_rows"], 486826)
        self.assertEqual(
            assessment["teacher_action_rows_source"],
            "teacher_summary.teacher_action_distribution.non_skip",
        )
        self.assertEqual(assessment["status"], "sufficient")


if __name__ == "__main__":
    unittest.main()
