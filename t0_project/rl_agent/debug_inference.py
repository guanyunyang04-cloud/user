import os
import sys
import argparse
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

# Import Env and custom Extractor
from lstm_extractor import CNNFeatureExtractor
from rl_utils import (
    FEATURE_COLS,
    load_csv,
    preprocess_data,
    train_eval_split,
    fit_standardizer,
    apply_standardizer,
    make_env,
    load_vecnormalize,
    unwrap_env,
    vec_reset,
    vec_step,
    extract_action,
    run_backtest
)

def load_real_data(csv_path):
    print(f"[*] Loading data: {csv_path}")
    return load_csv(csv_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Debug inference for RecurrentPPO.")
    parser.add_argument("--data-path", default=None, help="Path to CSV data file.")
    parser.add_argument("--model-path", default=None, help="Path to model .zip")
    parser.add_argument("--vecnormalize-path", default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--initial-total-asset", type=float, default=100000.0)
    parser.add_argument("--target-base-value", type=float, default=50000.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--zero-cost-mode", dest="zero_cost_mode", action="store_true", default=False)
    parser.add_argument("--dynamic-position", dest="dynamic_position", action="store_true", default=False)
    parser.add_argument("--reward-scale", type=float, default=5000.0)
    parser.add_argument("--continuous-action", dest="continuous_action", action="store_true", default=False)
    parser.add_argument("--initial-balance", type=float, default=None)
    parser.add_argument("--base-inventory", type=int, default=None)
    parser.add_argument("--base-slippage", type=float, default=0.0002)
    parser.add_argument("--impact-coeff", type=float, default=0.01)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--stamp-duty", type=float, default=0.0005)
    return parser.parse_args()

def debug_inference():
    print("==================================================")
    print(">>> Starting Inference Debugger <<<")
    print("==================================================")

    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')
    window_size = args.window_size
    initial_total_asset = args.initial_total_asset
    target_base_value = args.target_base_value
    data_path = args.data_path or os.path.join(base_dir, 'data', '600536.SH_1m_history.csv')
    if not os.path.exists(data_path):
        print(f"[-] Missing data file: {data_path}")
        return
        
    df_raw = load_real_data(data_path)
    df_raw = preprocess_data(df_raw)
    
    train_df_raw, raw_eval_df = train_eval_split(df_raw, train_ratio=args.train_ratio)

    scaler = fit_standardizer(train_df_raw, FEATURE_COLS)
    eval_df_features = apply_standardizer(raw_eval_df, scaler)

    def compute_initial_state(ref_price):
        lot_size = 100
        if args.base_inventory is None:
            base_inventory_local = int(target_base_value // (ref_price * lot_size)) * lot_size
        else:
            base_inventory_local = int(args.base_inventory // lot_size) * lot_size

        if args.initial_balance is None:
            initial_balance_local = initial_total_asset - (base_inventory_local * ref_price)
        else:
            initial_balance_local = float(args.initial_balance)

        implied_total_asset_local = initial_balance_local + (base_inventory_local * ref_price)
        if abs(implied_total_asset_local - initial_total_asset) > 1e-6:
            print(
                f"[!] initial_total_asset ({initial_total_asset:.2f}) != "
                f"balance+inventory value ({implied_total_asset_local:.2f}); using implied value."
            )
        return base_inventory_local, initial_balance_local, implied_total_asset_local

    eval_price_idx = window_size if len(raw_eval_df) > window_size else 0
    eval_price = float(raw_eval_df.iloc[eval_price_idx]['Close'])
    base_inventory, initial_balance, initial_total_asset = compute_initial_state(eval_price)

    base_env = make_env(
        eval_df_features,
        raw_eval_df,
        initial_balance=initial_balance,
        base_inventory=base_inventory,
        normalize=False,
        random_start=False,
        dynamic_position=args.dynamic_position,
        initial_total_asset=initial_total_asset,
        target_base_value=target_base_value,
        continuous_action=args.continuous_action,
        base_slippage=0.0 if args.zero_cost_mode else args.base_slippage,
        impact_coeff=0.0 if args.zero_cost_mode else args.impact_coeff,
        commission_rate=0.0 if args.zero_cost_mode else args.commission_rate,
        stamp_duty=0.0 if args.zero_cost_mode else args.stamp_duty,
        reward_scale=args.reward_scale,
    )
    vecnorm_path = args.vecnormalize_path or os.path.join(models_dir, 'vecnormalize.pkl')
    env = load_vecnormalize(vecnorm_path, base_env)
    
    model_path = args.model_path or os.path.join(models_dir, 'recurrent_ppo_tdx_t0_final.zip')
    if not os.path.exists(model_path):
        print(f"[-] Missing model: {model_path}")
        return

    if RecurrentPPO is None:
        print("[-] sb3-contrib is not installed. RecurrentPPO requires sb3-contrib.")
        print("    Install with: pip install sb3-contrib")
        return

    print(f"[*] Loading RecurrentPPO model...")
    custom_objects = {
        "features_extractor_class": CNNFeatureExtractor,
        "features_extractor_kwargs": dict(window_size=30, num_market_features=len(FEATURE_COLS))
    }
    model = RecurrentPPO.load(model_path, env=env, custom_objects=custom_objects, device="cpu")

    print("\n>>> Starting backtest (shared loop) <<<")
    backtest_summary = run_backtest(env, model, max_steps=(len(eval_df_features) - 31))

    # Robustly unwrap to the base env for account state
    inner_env = unwrap_env(env)
    initial_total_asset = inner_env.initial_total_asset

    print("\n==================================================")
    print(">>> Debug Summary Report <<<")
    print(f"Total steps in eval set: {len(eval_df_features) - 31}")
    print(f"Total successfully executed trades: {backtest_summary['trade_count']}")
    print(f"Initial Total Asset: {initial_total_asset:.4f}")
    if backtest_summary["final_asset"] is None:
        print("Final Total Asset (env): <missing> (no terminal info captured)")
        print("PnL Difference: <missing>")
    else:
        final_asset_snapshot = float(backtest_summary["final_asset"])
        print(f"Final Total Asset (env): {final_asset_snapshot:.4f}")
        print(f"PnL Difference: {final_asset_snapshot - initial_total_asset:+.4f}")
        if backtest_summary["total_pnl"] is not None:
            print(f"Total PnL (from env): {backtest_summary['total_pnl']:+.4f}")
        if backtest_summary["t0_pnl_total"] is not None:
            print(f"T0 Trading PnL (total): {backtest_summary['t0_pnl_total']:+.4f}")
        if backtest_summary["base_inventory_value_change"] is not None:
            print(f"Base Inventory Value Change: {backtest_summary['base_inventory_value_change']:+.4f}")
        if backtest_summary["avg_abs_action"] is not None:
            print(f"Avg |Action|: {backtest_summary['avg_abs_action']:.4f}")
    print("==================================================")
    
    if final_asset_snapshot is not None and final_asset_snapshot - initial_total_asset == 0.0:
        print("\n[!] Diagnostic: Net Profit is exactly 0.00")
        print("Check whether action execution happened, and confirm df_raw is correctly wired for PnL.")

if __name__ == "__main__":
    debug_inference()
    
