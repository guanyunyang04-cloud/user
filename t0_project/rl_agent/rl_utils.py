import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("[-] Please install stable-baselines3: pip install stable-baselines3[extra]")
    sys.exit(1)

FEATURE_COLS = [
    'Log_Return',
    'Close_Return',
    'Gap_Return',
    'OC_Return',
    'HL_Range',
    'ATR_14',
    'Volatility_20',
    'Volatility_60',
    'Momentum_10',
    'Momentum_30',
    'RSI_14',
    'MACD_Ratio',
    'MACD_Signal_Ratio',
    'EMA10_Bias',
    'EMA30_Bias',
    'Volume_Z',
    'Volume_Change',
    'BB_Width',
    'Time_Sin',
    'Time_Cos'
]


def preprocess_data(df):
    out = df.copy()
    
    # 1. Returns and price action (stationary)
    prev_close = out['Close'].shift(1).ffill()
    out['Log_Return'] = np.log((out['Close'] / prev_close).astype(float))
    out['Close_Return'] = (out['Close'] / prev_close) - 1.0
    out['Gap_Return'] = (out['Open'] / prev_close) - 1.0
    out['OC_Return'] = (out['Close'] - out['Open']) / prev_close
    out['HL_Range'] = (out['High'] - out['Low']) / prev_close

    # 2. Volatility and momentum
    out['Volatility_20'] = out['Log_Return'].rolling(window=20).std()
    out['Volatility_60'] = out['Log_Return'].rolling(window=60).std()
    out['Momentum_10'] = out['Log_Return'].rolling(window=10).mean()
    out['Momentum_30'] = out['Log_Return'].rolling(window=30).mean()

    # 3. RSI (14)
    delta = out['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-8)
    out['RSI_14'] = 100 - (100 / (1 + rs))

    # 4. MACD (Relative to Price)
    exp1 = out['Close'].ewm(span=12, adjust=False).mean()
    exp2 = out['Close'].ewm(span=26, adjust=False).mean()
    macd_raw = exp1 - exp2
    out['MACD_Ratio'] = macd_raw / out['Close']
    out['MACD_Signal_Ratio'] = out['MACD_Ratio'].ewm(span=9, adjust=False).mean()

    # 5. ATR (14)
    high_low = out['High'] - out['Low']
    high_prev = (out['High'] - prev_close).abs()
    low_prev = (out['Low'] - prev_close).abs()
    true_range = np.maximum(high_low, np.maximum(high_prev, low_prev))
    out['ATR_14'] = true_range.rolling(window=14).mean() / (prev_close + 1e-8)

    # 6. EMA Biases
    ema10 = out['Close'].ewm(span=10, adjust=False).mean()
    ema30 = out['Close'].ewm(span=30, adjust=False).mean()
    out['EMA10_Bias'] = (out['Close'] - ema10) / ema10
    out['EMA30_Bias'] = (out['Close'] - ema30) / ema30

    # 7. Volume features
    vol_mean = out['Volume'].rolling(window=20).mean()
    vol_std = out['Volume'].rolling(window=20).std()
    out['Volume_Z'] = (out['Volume'] - vol_mean) / (vol_std + 1e-8)
    prev_vol = out['Volume'].shift(1).ffill()
    out['Volume_Change'] = np.log((out['Volume'] / (prev_vol + 1e-8)).astype(float))

    # 8. Bollinger Bands width (volatility regime)
    bb_mid = out['Close'].rolling(window=20).mean()
    bb_std = out['Close'].rolling(window=20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    out['BB_Width'] = (bb_up - bb_low) / (bb_mid + 1e-8)

    # 9. Temporal Encoding (Sin/Cos of minute-of-day)
    bars_per_day = 240.0
    if 'Datetime' in out.columns:
        dt = pd.to_datetime(out['Datetime'], errors='coerce')
        hours = dt.dt.hour
        minutes = dt.dt.minute
        mins_passed = np.where(
            hours < 12,
            (hours - 9) * 60 + minutes - 30,
            120 + (hours - 13) * 60 + minutes
        )
        mins_passed = np.clip(mins_passed, 0, bars_per_day)
        minute_of_day = mins_passed
    else:
        minute_of_day = np.arange(len(out)) % bars_per_day

    angle = 2 * np.pi * (minute_of_day / bars_per_day)
    out['Time_Sin'] = np.sin(angle)
    out['Time_Cos'] = np.cos(angle)
    
    # Avoid look-ahead leakage: never backfill with future data
    out.ffill(inplace=True)
    out.fillna(0, inplace=True)
    
    # Replace infinities that might occur from log(0) or div by zero
    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    return out


def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path)


def train_eval_split(df_raw, train_ratio=0.8):
    split_idx = int(len(df_raw) * train_ratio)
    train_df_raw = df_raw.iloc[:split_idx].reset_index(drop=True)
    eval_df_raw = df_raw.iloc[split_idx:].reset_index(drop=True)
    return train_df_raw, eval_df_raw


def fit_standardizer(df, cols=FEATURE_COLS):
    stats = {}
    for c in cols:
        mean = df[c].mean()
        std = df[c].std() + 1e-8
        stats[c] = (mean, std)
    return stats


def apply_standardizer(df, stats):
    out = df.copy()
    for c, (mean, std) in stats.items():
        out[c] = (out[c] - mean) / std
    return out


def make_env(
    df_features,
    df_raw,
    initial_balance=100000.0,
    base_inventory=0,
    normalize=True,
    random_start=False,
    dynamic_position=False,
    initial_total_asset=100000.0,
    target_base_value=50000.0,
    continuous_action=True,
    base_slippage=0.0002,
    impact_coeff=0.01,
    max_slippage=0.005,
    commission_rate=0.0005,
    stamp_duty=0.0005,
    reward_scale=1000.0,
):
    # Local import to avoid circular import in some setups
    from tdx_t0_env import TDX_T0_Env

    env = DummyVecEnv([
        lambda: Monitor(TDX_T0_Env(
            df_features=df_features,
            df_raw=df_raw,
            initial_balance=initial_balance,
            base_inventory=base_inventory,
            feature_cols=FEATURE_COLS,
            random_start=random_start,
            dynamic_position=dynamic_position,
            initial_total_asset=initial_total_asset,
            target_base_value=target_base_value,
            continuous_action=continuous_action,
            base_slippage=base_slippage,
            impact_coeff=impact_coeff,
            max_slippage=max_slippage,
            commission_rate=commission_rate,
            stamp_duty=stamp_duty,
            reward_scale=reward_scale
        ))
    ])

    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def load_vecnormalize(path, base_env):
    if os.path.exists(path):
        env = VecNormalize.load(path, base_env)
        env.training = False
        env.norm_reward = False
        return env
    return base_env


def sync_vecnormalize(train_env, eval_env):
    # Copy running stats from train env to eval env
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def unwrap_env(env):
    # Unwrap VecNormalize/DummyVecEnv/Monitor to reach the base env
    while hasattr(env, 'venv'):
        env = env.venv
    if hasattr(env, 'envs'):
        env = env.envs[0]
    if hasattr(env, 'env'):  # Monitor wraps the real env
        env = env.env
    return env


def vec_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, (list, tuple)) else out


def vec_step(env, action):
    out = env.step(action)
    if isinstance(out, (list, tuple)) and len(out) == 4:
        obs, rewards, dones, infos = out
        return obs, rewards, dones, infos
    return out


def extract_action(action):
    if isinstance(action, (list, tuple, np.ndarray)):
        return int(action[0])
    return int(action)


def run_backtest(env, model, max_steps):
    """
    Shared backtest loop to keep train/debug results consistent.
    Returns summary dict with trades, final_asset, t0_pnl_total, base_inventory_value_change, total_pnl.
    """
    def _is_recurrent_model(m):
        policy = getattr(m, "policy", None)
        if policy is None:
            return False
        if hasattr(policy, "lstm_actor") or hasattr(policy, "lstm_critic") or hasattr(policy, "lstm"):
            return True
        return "Recurrent" in m.__class__.__name__

    def _supports_action_masks(m):
        sig = None
        try:
            import inspect
            sig = inspect.signature(m.predict)
        except Exception:
            return False
        return 'action_masks' in sig.parameters

    obs = vec_reset(env)
    trade_count = 0
    final_asset_snapshot = None
    t0_pnl_total = None
    last_base_inventory_value_change = None
    total_pnl = None
    action_abs_sum = 0.0
    action_steps = 0

    use_masks = isinstance(env.action_space, gym.spaces.Discrete)
    use_recurrent = _is_recurrent_model(model)
    allow_masks = (not use_recurrent) and use_masks and _supports_action_masks(model)
    state = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    for _ in range(max_steps):
        if use_recurrent:
            # RecurrentPPO does not support action masks; rely on env constraints/penalties.
            action, state = model.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=True
            )
        else:
            if allow_masks:
                action_masks = np.array(env.env_method("action_masks"))
                action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_step(env, action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        action_val = info.get('action')
        if action_val is not None:
            try:
                if isinstance(action_val, (list, tuple, np.ndarray)):
                    action_val = action_val[0]
                action_abs_sum += abs(float(action_val))
                action_steps += 1
            except Exception:
                pass

        if info.get('action_executed', False):
            trade_count += 1

        if info.get('total_asset') is not None:
            final_asset_snapshot = float(info.get('total_asset'))
        if info.get('t0_pnl_total') is not None:
            t0_pnl_total = float(info.get('t0_pnl_total'))
        if info.get('base_inventory_value_change') is not None:
            last_base_inventory_value_change = float(info.get('base_inventory_value_change'))
        if info.get('total_pnl') is not None:
            total_pnl = float(info.get('total_pnl'))

        done_flag = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        if use_recurrent:
            if isinstance(dones, (list, np.ndarray)):
                episode_start = np.array(dones, dtype=bool)
            else:
                episode_start = np.array([bool(dones)], dtype=bool)
        if done_flag:
            break

    return {
        "trade_count": trade_count,
        "final_asset": final_asset_snapshot,
        "t0_pnl_total": t0_pnl_total,
        "base_inventory_value_change": last_base_inventory_value_change,
        "total_pnl": total_pnl,
        "avg_abs_action": (action_abs_sum / action_steps) if action_steps > 0 else None,
    }
