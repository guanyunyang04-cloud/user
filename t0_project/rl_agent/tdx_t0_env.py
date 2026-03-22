import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TDX_T0_Env(gym.Env):
    """
    A-share T+0 intraday RL environment (Gymnasium).

    - df_features: standardized features for the policy (OHLCV).
    - df_raw: raw prices for PnL and fees.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df_features,
        df_raw,
        initial_balance=100000.0,
        base_inventory=0,
        window_size=30,
        feature_cols=None,
        bars_per_day=240,
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
        super().__init__()

        self.window_size = window_size
        self.bars_per_day = int(bars_per_day)
        self._day_index = None
        self._day_count = None
        self.random_start = bool(random_start)
        self.dynamic_position = bool(dynamic_position)
        self.initial_total_asset_target = float(initial_total_asset)
        self.target_base_value = float(target_base_value)
        self.continuous_action = bool(continuous_action)
        self.base_slippage = float(base_slippage)
        self.impact_coeff = float(impact_coeff)
        self.max_slippage = float(max_slippage)
        self._validate_inputs(df_features, df_raw)
        self.df_features = df_features.reset_index(drop=True)
        self.df_raw = df_raw.reset_index(drop=True)
        self._infer_bars_per_day(self.df_raw)

        self.max_steps = len(self.df_features) - 1

        # account state
        self.episode_initial_balance = initial_balance
        self.episode_base_inventory = base_inventory

        # transaction costs
        self.commission_rate = float(commission_rate)   # 0.05% (万5)
        self.stamp_duty = float(stamp_duty)        # 0.05% sell tax
        self.reward_scale = float(reward_scale)

        self.trade_size = 100

        if self.continuous_action:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

        if feature_cols is not None:
            self.feature_cols = feature_cols
        else:
            self.feature_cols = self.df_features.columns.tolist()
            if 'Datetime' in self.feature_cols:
                self.feature_cols.remove('Datetime')
                
        self.num_market_features = len(self.feature_cols)
        self.num_account_features = 4

        total_obs_size = (self.window_size * self.num_market_features) + self.num_account_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

    def _validate_inputs(self, df_features, df_raw):
        if df_features is None or df_raw is None:
            raise ValueError("df_features and df_raw must not be None")

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in required_cols:
            if c not in df_features.columns:
                raise ValueError(f"df_features missing column: {c}")
            if c not in df_raw.columns:
                raise ValueError(f"df_raw missing column: {c}")

        if len(df_features) != len(df_raw):
            raise ValueError("df_features and df_raw must have the same length")

        if len(df_features) <= (self.window_size + 2):
            raise ValueError("data length is too short for RL training")

    def _infer_bars_per_day(self, df_raw):
        """
        Infer bars-per-day from Datetime if available.
        Also precompute per-row index within day for robust time_left calc.
        """
        try:
            if 'Datetime' in df_raw.columns:
                dates = pd.to_datetime(df_raw['Datetime'], errors='coerce').dt.date
            elif isinstance(df_raw.index, pd.DatetimeIndex):
                dates = df_raw.index.date
            else:
                return

            dates_series = pd.Series(dates)
            counts = dates_series.value_counts(dropna=True)
            if counts.empty:
                return

            # Use the mode to handle partial days
            self.bars_per_day = int(counts.mode().iloc[0])

            # Precompute index within day and total bars for each row
            self._day_index = dates_series.groupby(dates_series).cumcount().to_numpy()
            count_map = counts.to_dict()
            self._day_count = dates_series.map(count_map).to_numpy()
        except Exception:
            # Fallback to provided bars_per_day
            self._day_index = None
            self._day_count = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start:
            # randomize starting point to avoid single long-trajectory overfitting
            max_start = max(self.window_size, self.max_steps - 1)
            if max_start <= self.window_size:
                self.current_step = self.window_size
            else:
                self.current_step = int(self.np_random.integers(self.window_size, max_start))
        else:
            self.current_step = self.window_size

        # Optionally recompute base position at the current start price
        if self.dynamic_position:
            current_price = self._get_current_price()
            lot_size = self.trade_size
            base_inventory = int(self.target_base_value // (current_price * lot_size)) * lot_size
            initial_balance = self.initial_total_asset_target - (base_inventory * current_price)
            self.episode_base_inventory = base_inventory
            self.episode_initial_balance = initial_balance
            self._base_entry_price = current_price
        else:
            # Use current price as entry reference for base inventory valuation
            self._base_entry_price = self._get_current_price()

        self.balance = self.episode_initial_balance
        self.inventory = self.episode_base_inventory
        
        # Intraday trackers
        self.daily_initial_balance = self.balance
        self.daily_initial_inventory = self.inventory
        self.today_bought_amount = 0
        self._last_trade_date = self._get_current_date()
        self._prev_t0_pnl = 0.0

        self.total_asset = self.balance + (self.inventory * self._get_current_price())
        self.initial_total_asset = self.total_asset

        return self._get_observation(), {}

    def _get_current_price(self):
        return float(self.df_raw.loc[self.current_step, 'Close'])

    def _get_current_volume(self):
        try:
            return float(self.df_raw.loc[self.current_step, 'Volume'])
        except Exception:
            return float(self.trade_size)

    def _calc_slippage(self, trade_shares, volume):
        vol = max(float(volume), 1.0)
        impact = self.impact_coeff * (float(trade_shares) / vol)
        slip = self.base_slippage + impact
        return float(min(self.max_slippage, max(0.0, slip)))

    def _is_force_close_time(self):
        if 'Datetime' in self.df_raw.columns:
            try:
                dt = pd.to_datetime(self.df_raw.loc[self.current_step, 'Datetime'])
                if dt.hour == 14 and dt.minute >= 55:
                    return True
            except Exception:
                pass
        # Fallback: last 5 bars of the day
        if self._day_index is not None and self._day_count is not None:
            try:
                idx_in_day = int(self._day_index[self.current_step])
                day_count = self._day_count[self.current_step]
                if day_count is not None and not np.isnan(day_count):
                    return idx_in_day >= (int(day_count) - 5)
            except Exception:
                pass
        if self.bars_per_day > 0:
            return (self.current_step % self.bars_per_day) >= (self.bars_per_day - 5)
        return False

    def _get_current_date(self):
        if 'Datetime' in self.df_raw.columns:
            try:
                return pd.to_datetime(self.df_raw.loc[self.current_step, 'Datetime']).date()
            except Exception:
                return None
        if isinstance(self.df_raw.index, pd.DatetimeIndex):
            return self.df_raw.index[self.current_step].date()
        # Fallback: derive a synthetic day index from bars_per_day
        if self.bars_per_day > 0:
            return int(self.current_step // self.bars_per_day)
        return None

    def _maybe_reset_intraday_state(self):
        forced_pnl = 0.0
        current_date = self._get_current_date()
        if current_date is None:
            return forced_pnl
        if self._last_trade_date is None:
            self._last_trade_date = current_date
            self.daily_initial_balance = self.balance
            self.daily_initial_inventory = self.inventory
            return forced_pnl
        if current_date != self._last_trade_date:
            # NEW DAY DETECTED
            # Force close any unclosed T+0 positions at today's opening price
            # to strictly penalize and prevent overnight carryover (swing trading).
            open_price = float(self.df_raw.loc[self.current_step, 'Open'])
            net_inventory = self.inventory - self.episode_base_inventory
            prev_balance = self.balance
            prev_inventory = self.inventory
            prev_close = open_price
            if self.current_step > 0:
                try:
                    prev_close = float(self.df_raw.loc[self.current_step - 1, 'Close'])
                except Exception:
                    prev_close = open_price
            slip = self._calc_slippage(abs(net_inventory), self._get_current_volume())
            
            if net_inventory > 0:
                # Force sell excess long positions
                revenue = net_inventory * open_price * (1 - slip)
                tax_fees = revenue * (self.commission_rate + self.stamp_duty)
                self.balance += (revenue - tax_fees)
            elif net_inventory < 0:
                # Force buy back missing short positions
                cost = abs(net_inventory) * open_price * (1 + slip)
                fees = cost * self.commission_rate
                self.balance -= (cost + fees)
            
            # Mathematically force inventory back to base
            self.inventory = self.episode_base_inventory

            # Capture forced-close PnL so reward can reflect it
            forced_cash_change = self.balance - prev_balance
            forced_pnl = forced_cash_change - (net_inventory * prev_close)

            # Reset daily trackers for the new day
            self.today_bought_amount = 0
            self.daily_initial_balance = self.balance
            self.daily_initial_inventory = self.inventory
            self._prev_t0_pnl = 0.0
            self._last_trade_date = current_date
        return forced_pnl

    def _get_time_left_today(self):
        # Prefer robust per-day bar counting if available
        if self._day_index is not None and self._day_count is not None:
            try:
                idx_in_day = int(self._day_index[self.current_step])
                day_count = self._day_count[self.current_step]
                if day_count is not None and not np.isnan(day_count) and day_count > 0:
                    time_left = (float(day_count) - float(idx_in_day) - 1.0) / float(day_count)
                    return float(max(0.0, min(1.0, time_left)))
            except Exception:
                pass

        # Fallback to clock-based estimate if Datetime exists
        if 'Datetime' in self.df_raw.columns:
            try:
                dt = pd.to_datetime(self.df_raw.loc[self.current_step, 'Datetime'])
                current_time = dt.time()
                if current_time.hour < 12:
                    mins_passed = (current_time.hour - 9) * 60 + current_time.minute - 30
                else:
                    mins_passed = 120 + (current_time.hour - 13) * 60 + current_time.minute
                mins_passed = max(0, min(240, mins_passed))
                time_left = (240 - mins_passed) / 240.0
                return float(time_left)
            except Exception:
                pass

        # Last-resort fallback when Datetime is missing
        if self.bars_per_day > 0:
            return float((self.bars_per_day - (self.current_step % self.bars_per_day)) / self.bars_per_day)
        return 0.0

    def _get_observation(self):
        market_data = self.df_features.loc[
            self.current_step - self.window_size : self.current_step - 1,
            self.feature_cols
        ].values

        market_data_flat = market_data.flatten()

        norm_balance = self.balance / self.episode_initial_balance
        if self.episode_base_inventory > 0:
            norm_inventory = self.inventory / self.episode_base_inventory
            norm_bought = self.today_bought_amount / self.episode_base_inventory
        else:
            # Avoid division by zero when starting with no inventory
            norm_inventory = 0.0
            norm_bought = 0.0
        time_left = self._get_time_left_today()

        account_data = np.array([norm_balance, norm_inventory, norm_bought, time_left])

        obs = np.concatenate((market_data_flat, account_data), axis=0).astype(np.float32)
        return obs

    def action_masks(self):
        """
        Returns a boolean array of shape (3,) indicating which actions are valid.
        Used when running discrete-action policies.
        0: Hold (Always True)
        1: Buy (True if enough balance)
        2: Sell (True if enough sellable T+1 inventory)
        """
        if self.continuous_action:
            return np.array([True])
        mask = np.array([True, True, True])
        
        # Check Buy
        current_price = self._get_current_price()
        slip = self._calc_slippage(self.trade_size, self._get_current_volume())
        cost = current_price * self.trade_size * (1 + slip)
        total_cost = cost * (1 + self.commission_rate)
        if self.balance < total_cost:
            mask[1] = False
            
        # Check Sell (T+1 rule)
        sellable_amount = self.inventory - self.today_bought_amount
        if sellable_amount < self.trade_size:
            mask[2] = False
            
        return mask

    def step(self, action):
        forced_pnl = self._maybe_reset_intraday_state()

        current_price = self._get_current_price()
        prev_total_asset = self.total_asset
        price_for_reward = current_price

        reward = 0.0
        trade_executed = False
        action_record = action
        if isinstance(action, (list, tuple, np.ndarray)):
            action_record = float(action[0])
        elif isinstance(action, np.generic):
            action_record = float(action)
        volume = self._get_current_volume()

        if self.continuous_action:
            action_value = action
            if isinstance(action, (list, tuple, np.ndarray)):
                action_value = float(action[0])
            action_value = float(np.clip(action_value, -1.0, 1.0))

            if action_value > 0:
                # Buy: fraction of max affordable shares
                max_affordable = int(
                    (self.balance // (current_price * (1 + self.base_slippage) * (1 + self.commission_rate)))
                    / self.trade_size
                ) * self.trade_size
                target_shares = int((max_affordable * action_value) // self.trade_size) * self.trade_size

                if target_shares <= 0:
                    if action_value > 0.05:
                        reward -= 0.1
                else:
                    slip = self._calc_slippage(target_shares, volume)
                    total_cost = current_price * target_shares * (1 + slip)
                    commission = total_cost * self.commission_rate
                    total_cost += commission

                    if self.balance < total_cost:
                        max_affordable = int(
                            (self.balance // (current_price * (1 + slip) * (1 + self.commission_rate)))
                            / self.trade_size
                        ) * self.trade_size
                        target_shares = min(target_shares, max_affordable)

                    if target_shares > 0:
                        slip = self._calc_slippage(target_shares, volume)
                        total_cost = current_price * target_shares * (1 + slip)
                        commission = total_cost * self.commission_rate
                        total_cost += commission
                        self.balance -= total_cost
                        self.inventory += target_shares
                        self.today_bought_amount += target_shares
                        trade_executed = True
                    else:
                        reward -= 0.1

            elif action_value < 0:
                # Sell: fraction of max sellable shares
                sellable_amount = self.inventory - self.today_bought_amount
                max_sell = int((sellable_amount // self.trade_size) * self.trade_size)
                target_shares = int((max_sell * (-action_value)) // self.trade_size) * self.trade_size

                if target_shares <= 0:
                    if action_value < -0.05:
                        reward -= 0.1
                else:
                    slip = self._calc_slippage(target_shares, volume)
                    revenue = current_price * target_shares * (1 - slip)
                    commission = revenue * self.commission_rate
                    tax = revenue * self.stamp_duty
                    total_revenue = revenue - commission - tax

                    self.balance += total_revenue
                    self.inventory -= target_shares
                    trade_executed = True
        else:
            if action == 1:  # buy
                slip = self._calc_slippage(self.trade_size, volume)
                cost = current_price * self.trade_size * (1 + slip)
                commission = cost * self.commission_rate
                total_cost = cost + commission

                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.inventory += self.trade_size
                    self.today_bought_amount += self.trade_size
                    trade_executed = True
                else:
                    reward -= 0.1

            elif action == 2:  # sell
                # T+1 constraint: only sell base inventory not bought today
                sellable_amount = self.inventory - self.today_bought_amount

                if sellable_amount >= self.trade_size:
                    slip = self._calc_slippage(self.trade_size, volume)
                    revenue = current_price * self.trade_size * (1 - slip)
                    commission = revenue * self.commission_rate
                    tax = revenue * self.stamp_duty
                    total_revenue = revenue - commission - tax

                    self.balance += total_revenue
                    self.inventory -= self.trade_size
                    trade_executed = True
                else:
                    reward -= 1.0

        # Force close near end of day (e.g., 14:55+)
        if self._is_force_close_time():
            net_inventory = self.inventory - self.episode_base_inventory
            if net_inventory != 0:
                slip = self._calc_slippage(abs(net_inventory), volume)
                if net_inventory > 0:
                    revenue = net_inventory * current_price * (1 - slip)
                    tax_fees = revenue * (self.commission_rate + self.stamp_duty)
                    self.balance += (revenue - tax_fees)
                else:
                    cost = abs(net_inventory) * current_price * (1 + slip)
                    fees = cost * self.commission_rate
                    self.balance -= (cost + fees)
                self.inventory = self.episode_base_inventory
                self.today_bought_amount = 0
                trade_executed = True
                # Use the forced-close execution price for reward/valuation (no look-ahead)
                price_for_reward = current_price

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Calculate T+0 net cashflow (Profit/Loss from today's T+0 trades, ignoring base inventory price changes)
        # We value the net change in inventory at current price to mark-to-market open T+0 positions
        net_inventory_change = self.inventory - self.daily_initial_inventory
        cash_change = self.balance - self.daily_initial_balance
        t0_net_pnl = cash_change + (net_inventory_change * price_for_reward)

        # Previous T+0 PnL tracking for step reward
        if not hasattr(self, '_prev_t0_pnl'):
            self._prev_t0_pnl = 0.0

        step_t0_pnl = t0_net_pnl - self._prev_t0_pnl
        self._prev_t0_pnl = t0_net_pnl

        # Reward is the scaled step PnL from trading activities, independent of base inventory beta
        if forced_pnl != 0.0:
            reward += (forced_pnl / self.episode_initial_balance) * self.reward_scale
        reward += (step_t0_pnl / self.episode_initial_balance) * self.reward_scale

        # Update total asset just for reporting
        # Use current bar price for valuation (no look-ahead)
        price_for_valuation = price_for_reward
        self.total_asset = self.balance + (self.inventory * price_for_valuation)
        base_inventory_value_change = self.episode_base_inventory * (price_for_valuation - self._base_entry_price)
        t0_pnl_total = (self.balance - self.episode_initial_balance) + (
            (self.inventory - self.episode_base_inventory) * price_for_valuation
        )
        total_pnl = self.total_asset - self.initial_total_asset

        if trade_executed:
            reward -= 0.05

        # Intraday overnight penalty: if not flat at end of day, penalize heavily
        current_time_left = self._get_time_left_today()
        
        # A true T+0 strategy should have inventory == episode_base_inventory at the end of the day.
        end_of_day_deviation = self.inventory - self.episode_base_inventory
        
        if current_time_left <= 0.02: # approx last 5 mins
            if end_of_day_deviation > 0:
                # Bought but didn't sell (Long T+0 not closed)
                reward -= 10.0
            elif end_of_day_deviation < 0:
                # Sold base inventory but didn't buy back (Short T+0 not closed)
                reward -= 10.0

        if terminated:
            # Final penalty for bankruptcy
            if self.balance < 0:
                reward -= 50.0

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'inventory': self.inventory,
            'total_asset': self.total_asset,
            'initial_total_asset': self.initial_total_asset,
            't0_net_pnl': t0_net_pnl,
            't0_pnl_total': t0_pnl_total,
            'base_inventory_value_change': base_inventory_value_change,
            'total_pnl': total_pnl,
            'action_executed': trade_executed,
            'action': action_record
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step} | Balance: {self.balance:.2f} | "
            f"Inventory: {self.inventory} | Total Asset: {self.total_asset:.2f}"
        )

# =====================================================================
# Minimal test stub
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(500) * 0.05) + 10.0
    mock_df = pd.DataFrame({
        'Open': closes + np.random.randn(500) * 0.01,
        'High': closes + 0.05,
        'Low': closes - 0.05,
        'Close': closes,
        'Volume': np.random.randint(1000, 5000, 500)
    })

    mock_features = mock_df.copy()
    for col in ['Open', 'High', 'Low', 'Close']:
        mock_features[col] = (mock_features[col] - mock_features['Close'].rolling(60).mean()) / (
            mock_features['Close'].rolling(60).std() + 1e-8
        )
    mock_features['Volume'] = (mock_features['Volume'] - mock_features['Volume'].rolling(60).mean()) / (
        mock_features['Volume'].rolling(60).std() + 1e-8
    )
    mock_features.fillna(0, inplace=True)

    env = TDX_T0_Env(df_features=mock_features, df_raw=mock_df)
    obs, info = env.reset()

    print("Env init ok. Obs shape:", obs.shape)

    total_reward = 0
    terminated = False

    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    env.render()
    print(f"Episode reward: {total_reward:.2f}")
