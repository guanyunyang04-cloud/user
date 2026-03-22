import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3 import PPO
except ImportError:
    print("[-] Please install required libs: pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from lstm_extractor import CNNFeatureExtractor
from rl_utils import (
    FEATURE_COLS,
    load_csv,
    preprocess_data,
    train_eval_split,
    fit_standardizer,
    apply_standardizer,
    make_env,
    sync_vecnormalize,
    vec_reset,
    vec_step,
    run_backtest
)

def load_real_data(csv_path):
    print(f"[*] Loading real TDX data from {csv_path}...")
    try:
        return load_csv(csv_path)
    except FileNotFoundError:
        print(f"[-] Data file {csv_path} not found!")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO for T+0 trading.")
    parser.add_argument("--data-path", default=None, help="Path to CSV data file.")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--initial-total-asset", type=float, default=100000.0)
    parser.add_argument("--target-base-value", type=float, default=50000.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=5000)

    parser.add_argument("--zero-cost-mode", dest="zero_cost_mode", action="store_true", default=False)
    parser.add_argument("--dynamic-position", dest="dynamic_position", action="store_true", default=False)
    parser.add_argument("--overfit-small-window", action="store_true", default=False)
    parser.add_argument("--overfit-bars", type=int, default=3000)
    parser.add_argument("--reward-scale", type=float, default=5000.0)
    parser.add_argument("--continuous-action", dest="continuous_action", action="store_true", default=False)

    parser.add_argument("--initial-balance", type=float, default=None)
    parser.add_argument("--base-inventory", type=int, default=None)

    parser.add_argument("--base-slippage", type=float, default=0.0002)
    parser.add_argument("--impact-coeff", type=float, default=0.01)
    parser.add_argument("--commission-rate", type=float, default=0.0005)
    parser.add_argument("--stamp-duty", type=float, default=0.0005)

    parser.add_argument("--model-name", default="recurrent_ppo_tdx_t0_final")
    parser.add_argument("--vecnormalize-name", default="vecnormalize.pkl")
    parser.add_argument("--load-model-path", default=None, help="Path to existing model .zip to continue training.")
    parser.add_argument("--load-vecnormalize-path", default=None, help="Path to existing VecNormalize .pkl to continue training.")
    parser.add_argument("--reset-num-timesteps", action="store_true", default=False)
    return parser.parse_args()


def main():
    print("============================================================")
    print(">>> Starting A-Share High-Frequency T+0 RL (RecurrentPPO) Pipeline <<<")
    print("============================================================")

    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    models_dir = os.path.join(project_root, 'models')
    logs_dir = os.path.join(project_root, 'logs')
    tb_dir = os.path.join(project_root, 'ppo_tdx_tensorboard')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    window_size = args.window_size
    initial_total_asset = args.initial_total_asset
    target_base_value = args.target_base_value

    # Load real historical data exported from TDX (1-minute data)
    data_path = args.data_path or os.path.join(base_dir, 'data', '600536.SH_1m_history.csv')
    df_raw = load_real_data(data_path)
    df_raw = preprocess_data(df_raw)

    if args.overfit_small_window and len(df_raw) > (args.overfit_bars + window_size + 2):
        df_raw = df_raw.iloc[:args.overfit_bars].reset_index(drop=True)
    
    train_df_raw, eval_df_raw = train_eval_split(df_raw, train_ratio=args.train_ratio)

    scaler = fit_standardizer(train_df_raw, FEATURE_COLS)
    train_df_features = apply_standardizer(train_df_raw, scaler)
    eval_df_features = apply_standardizer(eval_df_raw, scaler)

    def compute_initial_state(ref_price):
        # Assume we want to invest target_base_value in base inventory
        # inventory must be multiples of 100
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

    train_price_idx = window_size if len(train_df_raw) > window_size else 0
    eval_price_idx = window_size if len(eval_df_raw) > window_size else 0
    train_price = float(train_df_raw.iloc[train_price_idx]['Close'])
    eval_price = float(eval_df_raw.iloc[eval_price_idx]['Close'])
    train_base_inventory, train_initial_balance, train_initial_total_asset = compute_initial_state(train_price)
    eval_base_inventory, eval_initial_balance, eval_initial_total_asset = compute_initial_state(eval_price)

    print(
        f"[*] Train init: Balance={train_initial_balance:.2f}, "
        f"Inventory={train_base_inventory} @ Price={train_price:.2f}"
    )
    print(
        f"[*] Eval init:  Balance={eval_initial_balance:.2f}, "
        f"Inventory={eval_base_inventory} @ Price={eval_price:.2f}"
    )

    if args.zero_cost_mode:
        commission_rate = 0.0
        stamp_duty = 0.0
        base_slippage = 0.0
        impact_coeff = 0.0
    else:
        commission_rate = args.commission_rate
        stamp_duty = args.stamp_duty
        base_slippage = args.base_slippage
        impact_coeff = args.impact_coeff

    use_loaded_vecnorm = bool(args.load_vecnormalize_path)
    if use_loaded_vecnorm and not os.path.exists(args.load_vecnormalize_path):
        print(f"[-] VecNormalize file not found: {args.load_vecnormalize_path}")
        sys.exit(1)

    train_env = make_env(
        train_df_features,
        train_df_raw,
        initial_balance=train_initial_balance,
        base_inventory=train_base_inventory,
        normalize=not use_loaded_vecnorm,
        random_start=True,
        dynamic_position=args.dynamic_position,
        initial_total_asset=train_initial_total_asset,
        target_base_value=target_base_value,
        continuous_action=args.continuous_action,
        base_slippage=base_slippage,
        impact_coeff=impact_coeff,
        commission_rate=commission_rate,
        stamp_duty=stamp_duty,
        reward_scale=args.reward_scale,
    )

    eval_env = make_env(
        eval_df_features,
        eval_df_raw,
        initial_balance=eval_initial_balance,
        base_inventory=eval_base_inventory,
        normalize=not use_loaded_vecnorm,
        random_start=False,
        dynamic_position=args.dynamic_position,
        initial_total_asset=eval_initial_total_asset,
        target_base_value=target_base_value,
        continuous_action=args.continuous_action,
        base_slippage=base_slippage,
        impact_coeff=impact_coeff,
        commission_rate=commission_rate,
        stamp_duty=stamp_duty,
        reward_scale=args.reward_scale,
    )

    if use_loaded_vecnorm:
        train_env = VecNormalize.load(args.load_vecnormalize_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
        eval_env = VecNormalize.load(args.load_vecnormalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = sync_vecnormalize(train_env, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    if RecurrentPPO is None:
        print("[-] sb3-contrib is not installed. RecurrentPPO requires sb3-contrib.")
        print("    Install with: pip install sb3-contrib")
        sys.exit(1)

    # Use a non-recurrent feature extractor with RecurrentPPO
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(
            window_size=window_size, # default in TDX_T0_Env
            num_market_features=len(FEATURE_COLS) # Includes new alpha factors
        ),
    )

    if args.load_model_path:
        if not os.path.exists(args.load_model_path):
            print(f"[-] Model file not found: {args.load_model_path}")
            sys.exit(1)
        print(f"[*] Loading model for continued training: {args.load_model_path}")
        custom_objects = {
            "features_extractor_class": CNNFeatureExtractor,
            "features_extractor_kwargs": dict(window_size=window_size, num_market_features=len(FEATURE_COLS))
        }
        model = RecurrentPPO.load(args.load_model_path, env=train_env, custom_objects=custom_objects)
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir
        )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=logs_dir, 
        eval_freq=args.eval_freq,
        deterministic=True, 
        render=False
    )

    print(f"\n[*] Model built. Starting {args.total_timesteps} steps of training...")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=args.reset_num_timesteps
    )

    # Save VecNormalize statistics for later evaluation/inference
    vecnorm_path = os.path.join(models_dir, args.vecnormalize_name)
    train_env.save(vecnorm_path)

    save_path = os.path.join(models_dir, args.model_name)
    model.save(save_path)
    print(f"\n[+] Training complete! Model saved to {save_path}.zip")
    print(f"[+] VecNormalize stats saved to {vecnorm_path}")

    print("\n>>> Starting Backtest Inference on Test Set <<<")
    # Keep eval env deterministic
    eval_env.training = False
    eval_env.norm_reward = False
    backtest_summary = run_backtest(eval_env, model, max_steps=(len(eval_df_features) - 31))
    trade_count = backtest_summary["trade_count"]
    final_asset_snapshot = backtest_summary["final_asset"]
    t0_pnl_total = backtest_summary["t0_pnl_total"]
    last_base_inventory_value_change = backtest_summary["base_inventory_value_change"]
    total_pnl = backtest_summary["total_pnl"]
    avg_abs_action = backtest_summary["avg_abs_action"]
    print("========================================")
    print("Test Set Backtest Report (AI Independent Trading)")
    print(f"Total Trades (T+0 actions): {trade_count}")
    initial_total_asset = 100000.00
    # Pull initial_total_asset from env for consistency
    try:
        base_env = eval_env.venv.envs[0].env
        initial_total_asset = float(base_env.initial_total_asset)
    except Exception:
        pass
    print(f"Initial Total Asset: {initial_total_asset:.2f}")
    if final_asset_snapshot is None:
        final_asset_snapshot = initial_total_asset
    print(f"Final Total Asset: {final_asset_snapshot:.2f}")
    profit = final_asset_snapshot - initial_total_asset
    print(f"AI Net Profit: {profit:+.2f}")
    if total_pnl is not None:
        print(f"Total PnL (from env): {total_pnl:+.2f}")
    if t0_pnl_total is not None:
        print(f"T0 Trading PnL (total): {t0_pnl_total:+.2f}")
    if last_base_inventory_value_change is not None:
        print(f"Base Inventory Value Change: {last_base_inventory_value_change:+.2f}")
    if avg_abs_action is not None:
        print(f"Avg |Action|: {avg_abs_action:.4f}")
    print("========================================")

if __name__ == "__main__":
    main()
