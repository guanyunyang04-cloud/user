import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqcenter import tq
from execution import OrderManager, PaperBroker, SignalEvent, SignalType, build_execution_manager
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
import warnings
warnings.filterwarnings('ignore')


class _NoOpProgress:
    def update(self, n=1):
        pass

    def set_description_str(self, desc):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _create_progress(total, desc, unit="step", leave=True, position=0):
    if tqdm is None:
        return _NoOpProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
        ascii=True,
    )


def _iter_progress(iterable, total, desc, unit="item", leave=False, position=1):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
        ascii=True,
    )


def _progress_log(message):
    if tqdm is None:
        print(message)
    else:
        tqdm.write(str(message))

# =====================================================================
# 模块一：选股与波段主升段探测引擎 (来自 AdvancedShortTermStrategyV6)
# =====================================================================
class AdvancedShortTermStrategyV6:
    """结合 KAMA、威科夫量价行为、AO/AC 动能和均线结构的日线选股引擎。"""

    def __init__(self, block_name="沪深300", lookback_days=60, top_k=None,
                 target_tdx_block="AI威科夫龙头", market_risk_enabled=True, enable_logs=True):
        self.block_name = block_name
        self.lookback_days = lookback_days
        self.top_k = top_k
        self.target_tdx_block = target_tdx_block
        self.market_risk_enabled = market_risk_enabled
        self.enable_logs = enable_logs
        self.my_positions = {}

    def _log(self, message):
        if self.enable_logs:
            _progress_log(message)

    def _progress_iter(self, iterable, total, desc, unit="item", leave=False, position=1):
        if not self.enable_logs:
            return iterable
        return _iter_progress(iterable, total=total, desc=desc, unit=unit, leave=leave, position=position)

    def _passes_market_risk(self, index_closes, step_label=None):
        index_closes = index_closes.dropna()
        if len(index_closes) < 60:
            return True

        kama = self._calc_kama(index_closes).ewm(span=2, adjust=False).mean()
        current_close = float(index_closes.iloc[-1])
        current_kama = float(kama.iloc[-1])
        prev_kama = float(kama.iloc[-2])
        bull = index_closes.ewm(span=60, adjust=False).mean()
        bull_up = bool(bull.iloc[-1] >= bull.iloc[-2])
        is_safe = (current_close >= current_kama) or (current_kama >= prev_kama and bull_up)

        if step_label:
            self._log(f"{step_label} 上证最新: {current_close:.2f} | 市场KAMA: {current_kama:.2f}")
            if is_safe:
                self._log("[+] (大盘风控通过) 允许交易。")
            else:
                self._log("[-] (大盘风控拦截) 指数弱于 KAMA，暂停新开仓。")
        return is_safe

    def _check_market_risk_from_history(self, df_dict) -> bool:
        if not self.market_risk_enabled:
            return True
        if not df_dict or 'Close' not in df_dict or '999999.SH' not in df_dict['Close'].columns:
            return True
        return self._passes_market_risk(df_dict['Close']['999999.SH'])

    def _check_market_risk(self) -> bool:
        """基于上证指数 KAMA 的大盘风控。"""
        if not self.market_risk_enabled:
            self._log("[1/6] 已关闭大盘风控，跳过指数拦截。")
            return True

        self._log("[1/6] 正在执行全局大盘风控 (探测上证指数 KAMA 状态)...")
        market_index = '999999.SH'
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')

        try:
            df_dict = tq.get_market_data(
                field_list=['Close'],
                stock_list=[market_index],
                start_time=start_date,
                end_time=end_date,
                count=60,
                dividend_type='none',
                period='1d'
            )
            close_df = df_dict['Close']
            if close_df.empty or market_index not in close_df.columns:
                return True

            index_closes = close_df[market_index]
            return self._passes_market_risk(index_closes, step_label="      ")
        except Exception as e:
            self._log(f"      [!] 大盘风控计算异常: {e}")
            return False

    def _monitor_positions(self):
        pass

    @staticmethod
    def _calc_kama(series, er_window=10, fast=2, slow=30):
        direction = abs(series - series.shift(er_window))
        volatility = series.diff().abs().rolling(er_window).sum()
        er = direction / volatility.replace(0, np.nan)
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        smooth = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = np.zeros(len(series))
        seed_idx = min(er_window, len(series) - 1)
        kama[seed_idx] = series.iloc[seed_idx]
        for i in range(seed_idx + 1, len(series)):
            coeff = smooth.iloc[i]
            if pd.isna(coeff):
                coeff = slow_sc ** 2
            kama[i] = kama[i - 1] + coeff * (series.iloc[i] - kama[i - 1])
        return pd.Series(kama, index=series.index)

    @staticmethod
    def _cross_up(series_a, series_b):
        if len(series_a) < 2 or len(series_b) < 2:
            return False
        return bool(series_a.iloc[-2] <= series_b.iloc[-2] and series_a.iloc[-1] > series_b.iloc[-1])

    @staticmethod
    def _tdx_sma(series, n, m=1):
        alpha = m / float(n)
        return series.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _with_datetime_index(series):
        converted = series.copy()
        if not isinstance(converted.index, pd.DatetimeIndex):
            converted.index = pd.to_datetime(converted.index)
        return converted.sort_index()

    def _build_timeframe_bars(self, o, h, l, c, v, a, rule):
        frame = pd.DataFrame({
            'Open': self._with_datetime_index(o),
            'High': self._with_datetime_index(h),
            'Low': self._with_datetime_index(l),
            'Close': self._with_datetime_index(c),
            'Volume': self._with_datetime_index(v),
            'Amount': self._with_datetime_index(a),
        }).dropna(subset=['Close'])
        if frame.empty:
            return frame

        bars = frame.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Amount': 'sum',
        })
        return bars.dropna(subset=['Open', 'High', 'Low', 'Close'])

    def _higher_timeframe_bullish(self, o, h, l, c, v, a, rule):
        bars = self._build_timeframe_bars(o, h, l, c, v, a, rule)
        if len(bars) < 16:
            return False

        close_tf = bars['Close']
        kama_tf = self._calc_kama(close_tf).ewm(span=2, adjust=False).mean()
        if len(kama_tf.dropna()) < 2:
            return False

        upper_tf = self._tdx_sma(close_tf, 6.5, 1)
        lower_tf = self._tdx_sma(close_tf, 13.5, 1)
        return bool(
            close_tf.iloc[-1] > kama_tf.iloc[-1] and
            kama_tf.iloc[-1] >= kama_tf.iloc[-2] and
            upper_tf.iloc[-1] > lower_tf.iloc[-1]
        )

    def _get_universe(self):
        self._log("[2/6] 正在扫描全市场 A 股作为选股池...")
        try:
            all_stocks = tq.get_stock_list()
            if all_stocks and len(all_stocks) > 0:
                self._log(f"      成功获取全市场底层列表，共计 {len(all_stocks)} 只标的。")
                filtered_stocks = [s for s in all_stocks if s.startswith('60') or s.startswith('00')]
                self._log(f"      [智能过滤] 仅保留沪深主板，剩余 {len(filtered_stocks)} 只潜力股。")
                return filtered_stocks
        except Exception as e:
            self._log(f"      获取全市场股票失败: {e}，回退到默认样本。")
        return ['600000.SH', '600028.SH', '600030.SH', '600036.SH', '600050.SH']

    def _fetch_data(self, stocks):
        self._log("[3/6] 正在拉取日线数据 (包含大盘基准)...")
        fetch_count = max(self.lookback_days, 520)
        start_date = (datetime.now() - timedelta(days=fetch_count * 2)).strftime('%Y%m%d')
        end_date = datetime.now().strftime('%Y%m%d')
        all_to_fetch = list(set(stocks + ['999999.SH']))
        return tq.get_market_data(
            field_list=['Close', 'Volume', 'Amount', 'Open', 'High', 'Low'],
            stock_list=all_to_fetch,
            start_time=start_date,
            end_time=end_date,
            count=fetch_count,
            dividend_type='front',
            period='1d'
        )

    def _calculate_factors(self, df_dict):
        self._log("[4/6] 正在计算综合信号与多因子评分...")
        if not df_dict or 'Close' not in df_dict:
            return pd.DataFrame(columns=['Total_Score', 'Close', 'Signal_Tag'])

        c_df = df_dict['Close']
        v_df = df_dict['Volume']
        a_df = df_dict['Amount']
        o_df = df_dict['Open']
        h_df = df_dict['High']
        l_df = df_dict['Low']

        index_code = '999999.SH'
        if index_code in c_df.columns:
            idx_c = c_df[index_code].dropna()
            idx_ret20 = (idx_c - idx_c.shift(20)) / idx_c.shift(20).replace(0, np.nan)
            idx_ret60 = (idx_c - idx_c.shift(60)) / idx_c.shift(60).replace(0, np.nan)
        else:
            idx_ret20 = pd.Series(0.0, index=c_df.index)
            idx_ret60 = pd.Series(0.0, index=c_df.index)

        rows = []
        stock_iter = self._progress_iter(
            c_df.columns,
            total=len(c_df.columns),
            desc="      [4/6] 全市场逐只精算",
            unit="股",
            leave=False,
            position=1,
        )
        for stock in stock_iter:
            if stock == index_code:
                continue

            c = c_df[stock].dropna()
            v = v_df[stock].dropna()
            a = a_df[stock].dropna()
            o = o_df[stock].dropna()
            h = h_df[stock].dropna()
            l = l_df[stock].dropna()

            if len(c) < 200:
                continue

            try:
                kama = self._calc_kama(c).ewm(span=2, adjust=False).mean()
                bull = c.ewm(span=60, adjust=False).mean()
                long_regime = bool(c.iloc[-1] > kama.iloc[-1] and kama.iloc[-1] >= kama.iloc[-2] and bull.iloc[-1] >= bull.iloc[-2])
                short_regime = bool(c.iloc[-1] < kama.iloc[-1] and kama.iloc[-1] < kama.iloc[-2] and bull.iloc[-1] <= bull.iloc[-2])
                if short_regime:
                    continue

                daily_vwap = a / v.replace(0, np.nan)
                if (daily_vwap > c * 50).any():
                    daily_vwap = daily_vwap / 100
                cmid = daily_vwap.rolling(60, min_periods=20).mean()
                price_low_60 = l.rolling(60, min_periods=20).min()
                price_high_60 = h.rolling(60, min_periods=20).max()
                clow = price_low_60 + 0.05 * (price_high_60 - price_low_60)
                winp = (c - c.rolling(120, min_periods=20).min()) / (c.rolling(120, min_periods=20).max() - c.rolling(120, min_periods=20).min() + 1e-8) * 100
                vac = bool(winp.iloc[-1] > 80)

                tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                mid20 = c.ewm(span=20, adjust=False).mean()
                upper = mid20 + 2 * atr
                lower = mid20 - 2 * atr

                ma5 = v.rolling(5).mean()
                ma10 = v.rolling(10).mean()
                ma20 = v.rolling(20).mean()
                vrat = v / ma5.shift(1).replace(0, np.nan)
                double_volume = bool(v.iloc[-1] >= v.iloc[-2] * 1.9 and c.iloc[-1] > o.iloc[-1])
                vup = bool(v.iloc[-1] > ma5.iloc[-1] and v.iloc[-1] > ma10.iloc[-1] and vrat.iloc[-1] > 1.5)
                vdn = bool(v.iloc[-1] < ma10.iloc[-1] * 0.7)
                rvol = bool(v.iloc[-1] > ma20.iloc[-1] * 1.5 and vrat.iloc[-1] > 1.5)

                mid_price = (h + l) / 2
                ao = self._tdx_sma(mid_price, 5, 1) - self._tdx_sma(mid_price, 34, 1)
                ao_base = ao - self._tdx_sma(ao, 5, 1)
                ac = self._tdx_sma(ao_base, 5, 1)
                ao_up = bool(ao.iloc[-1] > ao.iloc[-2])
                ac_up = bool(ac.iloc[-1] > ac.iloc[-2])

                dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
                dea = dif.ewm(span=9, adjust=False).mean()
                mbar = (dif - dea) * 2
                maca = bool(mbar.iloc[-1] > mbar.iloc[-2] and dif.iloc[-1] > dif.iloc[-2])

                bias = (c - kama) / kama.replace(0, np.nan) * 100
                up_line = c.ewm(alpha=1 / 6.5, adjust=False).mean()
                down_line = c.ewm(alpha=1 / 13.5, adjust=False).mean()
                full_position_signal = self._cross_up(up_line, down_line)

                rsi1 = (
                    c.diff().clip(lower=0).ewm(alpha=1 / 6, adjust=False).mean() /
                    c.diff().abs().ewm(alpha=1 / 6, adjust=False).mean().replace(0, np.nan)
                ) * 100
                price_range = (h.iloc[-1] - l.iloc[-1]) + 1e-8
                lower_shadow = bool((min(o.iloc[-1], c.iloc[-1]) - l.iloc[-1]) / price_range > 0.6 and (h.iloc[-1] - l.iloc[-1]) / max(c.iloc[-2], 1e-8) > 0.03)
                ll20 = l.rolling(20).min()
                bottom_div = bool(l.iloc[-1] <= ll20.iloc[-1] and rsi1.iloc[-1] > rsi1.rolling(20).min().iloc[-2])
                plow = bool(l.iloc[-1] < lower.iloc[-1] and c.iloc[-1] > lower.iloc[-1])
                tbot = bool(lower_shadow and c.iloc[-1] < cmid.iloc[-1] and (bottom_div or plow) and vdn)
                xdd = bool(rsi1.iloc[-1] < 20 and c.iloc[-1] < clow.iloc[-1] and c.iloc[-1] < kama.iloc[-1])
                spring = bool(l.iloc[-1] <= ll20.shift(1).iloc[-1] and c.iloc[-1] > c.iloc[-2] and vdn and c.iloc[-1] < cmid.iloc[-1] and tbot)

                major_sup = self._cross_up(c, kama) and bull.iloc[-1] >= bull.iloc[-2] and vup and c.iloc[-1] > cmid.iloc[-1]
                major_run = self._cross_up(c, kama) and maca and bias.iloc[-1] < 5 and vup and c.iloc[-1] > cmid.iloc[-1] and bull.iloc[-1] >= bull.iloc[-2]
                major_buy = major_sup or major_run
                fund_break = self._cross_up(c, cmid.fillna(method='ffill')) and rvol and c.iloc[-1] > o.iloc[-1] and long_regime
                giant_vol = bool(v.iloc[-1] > ma10.iloc[-1] * 2 and v.iloc[-1] > v.iloc[-2])
                strong_body = bool((c.iloc[-1] - o.iloc[-1]) / max(o.iloc[-1], 1e-8) > 0.04 and c.iloc[-1] >= h.iloc[-1] * 0.98)
                sos_add = bool(giant_vol and strong_body and self._cross_up(c, upper.fillna(method='ffill')) and vac and long_regime)
                resonance = bool(double_volume and ao_up and ac_up and c.iloc[-1] > c.ewm(span=10, adjust=False).mean().iloc[-1])
                low_absorb = bool(long_regime and 0 < bias.iloc[-1] < 3 and c.iloc[-1] > o.iloc[-1] and c.iloc[-2] <= o.iloc[-2] and c.iloc[-1] > cmid.iloc[-1])
                red_wave = bool(full_position_signal)
                weekly_bullish = self._higher_timeframe_bullish(o, h, l, c, v, a, 'W-FRI')
                monthly_bullish = self._higher_timeframe_bullish(o, h, l, c, v, a, 'M')
                if not (weekly_bullish and monthly_bullish):
                    continue

                hh20 = h.rolling(20).max()
                touch_high = bool(h.iloc[-1] >= hh20.iloc[-1] and h.iloc[-2] < hh20.shift(1).iloc[-1])
                hot = bool(bias.iloc[-1] > 12 or rsi1.iloc[-1] > 80)
                trim_signal = bool(long_regime and (touch_high or hot))
                if trim_signal:
                    continue
                if sos_add and hot:
                    continue

                trigger_group_1 = bool(major_buy or fund_break or sos_add)
                trigger_group_2 = bool(resonance)
                trigger_group_3 = bool(red_wave)
                trigger_count = int(trigger_group_1) + int(trigger_group_2) + int(trigger_group_3)
                if trigger_count < 2:
                    continue

                angle = (kama - kama.shift(1)) / kama.shift(1).replace(0, np.nan) * 1000
                alpha20 = (c - c.shift(20)) / c.shift(20).replace(0, np.nan) - idx_ret20.reindex(c.index).fillna(0.0)
                alpha60 = (c - c.shift(60)) / c.shift(60).replace(0, np.nan) - idx_ret60.reindex(c.index).fillna(0.0)
                m_fast = c.ewm(span=9, adjust=False).mean() - c.ewm(span=150, adjust=False).mean()
                m_slow = m_fast.ewm(span=12, adjust=False).mean()
                m_strength = (m_fast - m_slow) / atr.replace(0, np.nan) * 10
                day_rev = (c - l) / (h - l + 1e-8) * 10

                score = 0.0
                score += 18 if long_regime else -10
                score += 24 if major_buy else 0
                score += 16 if fund_break else 0
                score += 22 if sos_add else 0
                score += 10 if resonance else 0
                score += 18 if low_absorb else 0
                score += 8 if (xdd or spring) else 0
                score += 6 if vac else 0
                score += 8 if full_position_signal else -2
                score += min(max(vrat.iloc[-1] - 1.0, 0.0), 3.0) * 4
                score += min(max(angle.iloc[-1], -5.0), 15.0) * 1.5
                score += min(max(m_strength.iloc[-1], -5.0), 12.0) * 1.8
                score += min(max(alpha20.iloc[-1] * 100, -8.0), 20.0) * 1.6
                score += min(max(alpha60.iloc[-1] * 100, -12.0), 30.0) * 0.9
                score += min(max(day_rev.iloc[-1], 0.0), 10.0)
                score += max(0.0, 8.0 - abs(bias.iloc[-1] - 3.0))

                tags = []
                if major_buy:
                    tags.append('主升起航')
                if fund_break:
                    tags.append('资金突破')
                if sos_add:
                    tags.append('SOS加仓')
                if resonance:
                    tags.append('强共振')
                if xdd or spring:
                    tags.append('极值摸底')
                if low_absorb:
                    tags.append('回踩吸筹')
                if red_wave:
                    tags.append('红色波段')
                if weekly_bullish:
                    tags.append('周线多头')
                if monthly_bullish:
                    tags.append('月线多头')

                rows.append({
                    'stock': stock,
                    'Total_Score': float(score),
                    'Close': float(c.iloc[-1]),
                    'Signal_Tag': ' / '.join(tags),
                })
            except Exception as e:
                self._log(f"      计算 {stock} 因子异常: {e}")

        if not rows:
            return pd.DataFrame(columns=['Total_Score', 'Close', 'Signal_Tag'])
        return pd.DataFrame(rows).set_index('stock')

    def _score_and_rank(self, factors_df):
        self._log("[5/6] 正在对候选标的排序...")
        if factors_df.empty:
            self._log("      [!] 没有候选股票通过筛选条件。")
            return factors_df

        ranked_df = factors_df.replace([np.inf, -np.inf], np.nan)
        ranked_df = ranked_df.dropna(subset=['Total_Score', 'Close'])
        ranked_df = ranked_df.sort_values('Total_Score', ascending=False)

        if self.top_k is not None and self.top_k > 0:
            ranked_df = ranked_df.head(self.top_k)

        self._log(f"      排名完成，最终入选 {len(ranked_df)} 只标的。")
        return ranked_df

    def _execute_signals(self, ranked_df):
        self._log("[6/6] 正在推送信号到通达信自定义板块...")
        self.my_positions = {}
        if ranked_df.empty:
            self._log("没有股票满足筛选条件。")
            return

        target_stocks = ranked_df.index.tolist()
        self.my_positions = ranked_df['Close'].to_dict()
        self._log(ranked_df[['Total_Score', 'Signal_Tag']].head(20).to_string())

        tq.create_sector(block_code='AIWyckoff', block_name=self.target_tdx_block)
        tq.send_user_block(block_code='AIWyckoff', stocks=[])
        tq.send_user_block(block_code='AIWyckoff', stocks=target_stocks, show=True)

    def select_from_history(self, df_dict, execute_signals=False):
        self.my_positions = {}
        if not self._check_market_risk_from_history(df_dict):
            return pd.DataFrame(columns=['Total_Score', 'Close', 'Signal_Tag'])

        factors_df = self._calculate_factors(df_dict)
        ranked_df = self._score_and_rank(factors_df)
        self.my_positions = ranked_df['Close'].to_dict() if not ranked_df.empty else {}
        if execute_signals:
            self._execute_signals(ranked_df)
        return ranked_df

    def run(self):
        try:
            progress_bar = _create_progress(total=6, desc="选股流程", unit="步", leave=True, position=0) if self.enable_logs else _NoOpProgress()
            with progress_bar as progress:
                progress.set_description_str("选股流程 1/6 大盘风控")
                if not self._check_market_risk():
                    progress.update(1)
                    return False
                progress.update(1)

                progress.set_description_str("选股流程 2/6 股票池扫描")
                stocks = self._get_universe()
                progress.update(1)

                progress.set_description_str("选股流程 3/6 日线数据拉取")
                df_dict = self._fetch_data(stocks)
                progress.update(1)

                progress.set_description_str("选股流程 4/6 全市场精算")
                factors_df = self._calculate_factors(df_dict)
                progress.update(1)

                progress.set_description_str("选股流程 5/6 候选排序")
                ranked_df = self._score_and_rank(factors_df)
                progress.update(1)

                progress.set_description_str("选股流程 6/6 信号推送")
                self._execute_signals(ranked_df)
                progress.update(1)
            return True
        except Exception as e:
            self._log(f"选股策略运行期间发生错误: {e}")
            return False
# =====================================================================
# 模块二：日内高频监控与做T引擎 (重构为单次检查架构)
# =====================================================================
class IntradayRealtimeBotV2:
    """
    机构级盘中实时监控与做T引擎 V2.0 (融合AI量化微观结构与强化学习风控特征版)
    """
    def __init__(self, positions=None, execution_manager=None, exit_rules=None, sizing_rules=None):
        self.my_positions = positions if positions else {}
        self.execution_manager = execution_manager or OrderManager(PaperBroker(initial_cash=1_000_000.0))
        self.exit_rules = {
            'hard_stop_loss_pct': 0.08,
            'tp1_pct': 0.10,
            'tp1_sell_ratio': 0.33,
            'tp2_pct': 0.18,
            'tp2_sell_ratio': 0.50,
            'trail_reduce_drawdown_pct': 0.06,
            'trail_reduce_min_profit_pct': 0.08,
            'trail_reduce_sell_ratio': 0.50,
            'trail_clear_drawdown_pct': 0.10,
            'sell_if_out_of_pool': False,
            'out_of_pool_clear_rounds': 2,
        }
        if exit_rules:
            self.exit_rules.update(exit_rules)
        self.sizing_rules = {
            'entry_position_ratio': 0.10,
            'add_position_ratio': 0.05,
            'min_trade_lot': 100,
        }
        if sizing_rules:
            self.sizing_rules.update(sizing_rules)
        if isinstance(self.execution_manager.broker, PaperBroker):
            for stock, cost in self.my_positions.items():
                if cost is not None:
                    self.execution_manager.broker.seed_position(stock, 100, float(cost))
        # 扩展状态空间 (MDP State): 记录单日交易次数、最高收益(用于移动止盈)、是否已清仓等
        self.trade_state = {
            stock: self._build_trade_state(self.my_positions.get(stock))
            for stock in self.my_positions
        }
        self.history_buffer = {stock: pd.DataFrame() for stock in self.my_positions}

    def _build_trade_state(self, cost):
        has_position = cost is not None
        initial_price = float(cost) if has_position else 0.0
        return {
            'last_alert': 0,
            'last_buy_signal': 0,
            'last_sell_signal': 0,
            'has_position': has_position,
            'selected_price': cost,
            'position_qty': 100 if has_position else 0,
            'cleared': False,
            'trade_count': 0,
            'max_profit': 0.0,
            'win_rate_proxy': 0.60,
            'peak_price': initial_price,
            'tp1_done': False,
            'tp2_done': False,
            'trail_reduced': False,
            'pool_reduced': False,
            'out_of_pool_rounds': 0,
        }

    def add_stock(self, stock, cost):
        self.my_positions[stock] = cost
        self.trade_state[stock] = self._build_trade_state(cost)
        self.history_buffer[stock] = pd.DataFrame()
        if cost is not None and isinstance(self.execution_manager.broker, PaperBroker):
            self.execution_manager.broker.seed_position(stock, 100, float(cost))

    def remove_stock(self, stock):
        if stock in self.my_positions:
            del self.my_positions[stock]
        if stock in self.trade_state:
            del self.trade_state[stock]
        if stock in self.history_buffer:
            del self.history_buffer[stock]

    def _send_alert(self, stock, price, bs_flag, reason, volume="100"):
        now_str = datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            tq.send_warn(
                stock_list=[stock], time_list=[now_str],
                price_list=[str(price)], close_list=[str(price)],
                volum_list=[volume], bs_flag_list=[str(bs_flag)], 
                warn_type_list=['0'], reason_list=[reason], count=1
            )
            print(f"[{now_str[8:10]}:{now_str[10:12]}:{now_str[12:14]}] [!] {stock} 触发: {reason} | 现价: {price:.2f} | 建议股数: {volume}")
        except Exception as e:
            pass

    def _sync_position_state(self, stock):
        state = self.trade_state.get(stock)
        if state is None:
            return

        position = self.execution_manager.broker.get_positions().get(stock)
        if position is None or position.quantity <= 0:
            self.my_positions[stock] = None
            state['has_position'] = False
            state['position_qty'] = 0
            state['peak_price'] = 0.0
            return

        self.my_positions[stock] = float(position.cost_price)
        state['has_position'] = True
        state['position_qty'] = int(position.quantity)
        state['cleared'] = False
        state['selected_price'] = float(position.cost_price)
        state['peak_price'] = max(float(state.get('peak_price', 0.0)), float(position.last_price), float(position.cost_price))

    @staticmethod
    def _calculate_reduce_quantity(position_qty, ratio):
        position_qty = max(int(position_qty), 0)
        if position_qty <= 0:
            return 0
        raw_qty = max(position_qty * float(ratio), 0.0)
        rounded_qty = int(raw_qty / 100) * 100
        if rounded_qty <= 0:
            rounded_qty = 100 if position_qty >= 100 else position_qty
        return min(rounded_qty, position_qty)

    def _submit_reduce(self, stock, price, reason, ratio):
        state = self.trade_state.get(stock, {})
        qty = self._calculate_reduce_quantity(state.get('position_qty', 0), ratio)
        if qty <= 0:
            return False
        return self._submit_signal(stock, SignalType.SELL_REDUCE, price, reason, default_qty=qty)

    def _submit_clear(self, stock, price, reason):
        state = self.trade_state.get(stock, {})
        qty = max(int(state.get('position_qty', 0)), 0)
        if qty <= 0:
            return False
        return self._submit_signal(stock, SignalType.SELL_EXIT, price, reason, default_qty=qty)

    def _calculate_target_buy_quantity(self, stock, price, signal_type):
        if price <= 0:
            return 0

        account = self.execution_manager.broker.get_account()
        positions = self.execution_manager.broker.get_positions()
        current_position = positions.get(stock)
        current_qty = int(current_position.quantity) if current_position else 0
        current_value = current_qty * float(price)
        min_lot = max(int(self.sizing_rules.get('min_trade_lot', 100)), 1)

        if signal_type == SignalType.BUY_ENTRY:
            target_ratio = float(self.sizing_rules.get('entry_position_ratio', 0.10))
            target_value = max(account.total_asset * target_ratio - current_value, 0.0)
        else:
            target_ratio = float(self.sizing_rules.get('add_position_ratio', 0.05))
            target_value = account.total_asset * target_ratio

        max_affordable_value = min(target_value, account.available_cash)
        raw_qty = int(max_affordable_value // max(float(price), 1e-8))
        sized_qty = (raw_qty // min_lot) * min_lot
        if sized_qty < min_lot:
            return 0
        return sized_qty

    def refresh_candidate_pool(self, latest_candidate_set):
        latest_candidate_set = set(latest_candidate_set or [])
        now_ts = time.time()

        for stock in list(self.my_positions.keys()):
            state = self.trade_state.get(stock)
            if state is None:
                continue

            if stock in latest_candidate_set:
                state['out_of_pool_rounds'] = 0
                state['pool_reduced'] = False
                continue

            if not state.get('has_position', False):
                state['cleared'] = True
                continue

            state['out_of_pool_rounds'] = int(state.get('out_of_pool_rounds', 0)) + 1
            if not self.exit_rules.get('sell_if_out_of_pool', False):
                continue

            try:
                snap = tq.get_market_snapshot(stock_code=stock)
                now_price = float(snap.get('Now', 0)) if snap and snap.get('ErrorId') == '0' else 0.0
            except Exception:
                now_price = 0.0
            if now_price <= 0:
                now_price = float(self.my_positions.get(stock) or state.get('selected_price') or 0.0)
            if now_price <= 0:
                continue

            if not state.get('pool_reduced', False):
                if self._submit_reduce(stock, now_price, "【调出股票池】先减仓观察", ratio=0.50):
                    state['pool_reduced'] = True
                    state['last_alert'] = now_ts
                    state['last_sell_signal'] = now_ts
                continue

            if state.get('out_of_pool_rounds', 0) >= int(self.exit_rules.get('out_of_pool_clear_rounds', 2)):
                if self._submit_clear(stock, now_price, "【连续调出股票池】执行清仓"):
                    state['last_alert'] = now_ts
                    state['last_sell_signal'] = now_ts

    def _submit_signal(self, stock, signal_type, price, reason, default_qty=100):
        state = self.trade_state.get(stock)
        if state is None:
            return False

        suggested_qty = default_qty
        if signal_type in {SignalType.BUY_ENTRY, SignalType.BUY_ADD}:
            computed_qty = self._calculate_target_buy_quantity(stock, float(price), signal_type)
            suggested_qty = computed_qty
        elif signal_type == SignalType.SELL_EXIT:
            suggested_qty = max(int(state.get('position_qty', 0)), 0)
        elif signal_type == SignalType.SELL_REDUCE:
            suggested_qty = min(default_qty, max(int(state.get('position_qty', 0)), 0))

        if suggested_qty <= 0:
            return False

        signal = SignalEvent(
            stock=stock,
            signal_type=signal_type,
            signal_price=float(price),
            signal_time=datetime.now().strftime('%Y%m%d%H%M%S'),
            reason=reason,
            suggested_qty=int(suggested_qty),
        )
        try:
            record = self.execution_manager.process_signal(signal)
        except NotImplementedError as exc:
            print(f"      [执行层未实现] {stock} {reason}: {exc}")
            return False
        if record is None:
            print(f"      [执行层拦截] {stock} {reason} 未通过风控或无可用头寸。")
            return False

        self._sync_position_state(stock)
        position = self.execution_manager.broker.get_positions().get(stock)
        qty_text = str(record.filled_quantity)
        if signal_type in {SignalType.SELL_REDUCE, SignalType.SELL_EXIT} and (position is None or position.quantity <= 0):
            state['cleared'] = True
            self.my_positions[stock] = None
        self._send_alert(stock, record.avg_fill_price, 0 if record.request.side.value == 'buy' else 1, f"{reason}[模拟成交]", qty_text)
        return True
            
    def _calculate_kelly_fraction(self, win_rate, win_loss_ratio=1.5):
        """动态凯利公式 (Kelly Criterion) 计算仓位，采用半凯利防范回撤"""
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        return max(0, kelly_pct / 2) # 半凯利

    @staticmethod
    def _cross_up_series(series_a, series_b):
        return bool(series_a.iloc[-1] > series_b.iloc[-1] and series_a.iloc[-2] <= series_b.iloc[-2])

    @staticmethod
    def _cross_down_series(series_a, series_b):
        return bool(series_a.iloc[-1] < series_b.iloc[-1] and series_a.iloc[-2] >= series_b.iloc[-2])

    @staticmethod
    def _trading_minutes_passed(now_dt):
        morning_open = now_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        morning_close = now_dt.replace(hour=11, minute=30, second=0, microsecond=0)
        afternoon_open = now_dt.replace(hour=13, minute=0, second=0, microsecond=0)

        if now_dt <= morning_open:
            return 0.0
        if now_dt <= morning_close:
            return max(0.0, (now_dt - morning_open).total_seconds() / 60.0)
        if now_dt < afternoon_open:
            return 120.0
        return 120.0 + max(0.0, (now_dt - afternoon_open).total_seconds() / 60.0)

    def _load_intraday_bars(self, stock, count=120):
        df_dict = tq.get_market_data(
            field_list=['Close', 'Volume', 'Amount', 'Open', 'High', 'Low'],
            stock_list=[stock],
            start_time='',
            end_time='',
            count=count,
            dividend_type='front',
            period='1m'
        )
        if 'Close' not in df_dict or df_dict['Close'].empty or stock not in df_dict['Close'].columns:
            return pd.DataFrame()

        df = pd.DataFrame({
            'Close': df_dict['Close'][stock],
            'Volume': df_dict['Volume'][stock],
            'Amount': df_dict['Amount'][stock],
            'Open': df_dict['Open'][stock],
            'High': df_dict['High'][stock],
            'Low': df_dict['Low'][stock],
        }).dropna(subset=['Close'])

        if df.empty:
            return df

        try:
            dt_index = pd.to_datetime(df.index)
            today = datetime.now().date()
            df.index = dt_index
            df = df[df.index.date == today]
        except Exception:
            pass

        return df

    def check_once(self):
        """执行一次全池盘口快照轮询，深度集成通达信分时高抛低吸逻辑"""
        if not self.my_positions:
            return
            
        for stock, cost in self.my_positions.items():
            state = self.trade_state[stock]
            if state['cleared']: continue
                
            try:
                snap = tq.get_market_snapshot(stock_code=stock)
                if not snap or snap.get('ErrorId') != '0': continue
                
                now_price = float(snap.get('Now', 0))
                vwap = float(snap.get('Average', 0))
                
                if now_price == 0 or vwap == 0: continue

                now_dt = datetime.now()
                minutes_passed = self._trading_minutes_passed(now_dt)
                df = self._load_intraday_bars(stock, count=120)
                if df.empty or len(df) < 15:
                    continue
                self.history_buffer[stock] = df

                c_series = df['Close']
                v_series = df['Volume']
                a_series = df['Amount']
                open_filter = minutes_passed > 5

                cum_volume = v_series.replace(0, np.nan).cumsum()
                vwap_series = a_series.cumsum() / cum_volume
                if (vwap_series > c_series * 50).any():
                    vwap_series = vwap_series / 100
                vwap_series = vwap_series.fillna(method='ffill').fillna(vwap)
                vwap = float(vwap_series.iloc[-1])

                bar_count = len(df)
                minute_bars = pd.Series(np.arange(1, bar_count + 1), index=df.index, dtype=float)
                estimated_amp = 0.015
                transition_period = 30.0
                initial_upper = vwap_series * (1 + estimated_amp)
                initial_lower = vwap_series * (1 - estimated_amp)
                real_deviation = (c_series - vwap_series).abs().ewm(span=20, adjust=False).mean()
                multiplier = pd.Series(
                    np.where(
                        minute_bars < transition_period,
                        2.0 + 1.0 * (1.0 - minute_bars / transition_period),
                        2.0
                    ),
                    index=df.index,
                    dtype=float,
                )
                dynamic_upper = vwap_series + multiplier * real_deviation
                dynamic_lower = vwap_series - multiplier * real_deviation
                transition_ratio = np.minimum(minute_bars / transition_period, 1.0)
                upper_band_series = initial_upper * (1.0 - transition_ratio) + dynamic_upper * transition_ratio
                lower_band_series = initial_lower * (1.0 - transition_ratio) + dynamic_lower * transition_ratio
                upper_band = float(upper_band_series.iloc[-1])
                lower_band = float(lower_band_series.iloc[-1])

                ema6 = c_series.ewm(span=6, adjust=False).mean()
                ema13 = c_series.ewm(span=13, adjust=False).mean()
                dif = ema6 - ema13
                dea = dif.ewm(span=5, adjust=False).mean()
                cross_up = self._cross_up_series(dif, dea)
                cross_down = self._cross_down_series(dif, dea)
                dif_up_turn = (dif.iloc[-1] > dif.iloc[-2]) and (dif.iloc[-2] <= dif.iloc[-3])
                dif_down_turn = (dif.iloc[-1] < dif.iloc[-2]) and (dif.iloc[-2] >= dif.iloc[-3])

                near_lower = bool((c_series.tail(5) <= lower_band_series.tail(5) * 1.001).any())
                near_upper = bool((c_series.tail(5) >= upper_band_series.tail(5) * 0.999).any())

                current_timestamp = time.time()
                v_ma5 = v_series.rolling(5).mean()
                urgent_pull = bool(
                    c_series.iloc[-1] > c_series.iloc[-2] and
                    c_series.iloc[-1] / max(c_series.iloc[-2], 1e-8) > 1.008 and
                    v_series.iloc[-1] > v_ma5.iloc[-1] * 1.5 and
                    open_filter
                )
                urgent_drop = bool(
                    c_series.iloc[-1] < c_series.iloc[-2] and
                    c_series.iloc[-1] / max(c_series.iloc[-2], 1e-8) < 0.992 and
                    v_series.iloc[-1] > v_ma5.iloc[-1] * 1.5 and
                    open_filter
                )
                has_position = bool(state.get('has_position', False))

                if not has_position:
                    if urgent_pull and current_timestamp - state['last_alert'] >= 300:
                        if self._submit_signal(stock, SignalType.BUY_ENTRY, now_price, "↖抢筹建仓", default_qty=100):
                            state['selected_price'] = now_price
                            state['max_profit'] = 0.0
                            state['peak_price'] = now_price
                            state['tp1_done'] = False
                            state['tp2_done'] = False
                            state['trail_reduced'] = False
                            state['pool_reduced'] = False
                            state['out_of_pool_rounds'] = 0
                            state['last_alert'] = current_timestamp
                            state['last_buy_signal'] = current_timestamp
                            state['trade_count'] += 1
                    continue

                cost = self.my_positions.get(stock) or state.get('selected_price')
                if cost is None or cost <= 0:
                    continue
                profit = (now_price - cost) / cost
                if profit > state['max_profit']:
                    state['max_profit'] = profit
                state['peak_price'] = max(float(state.get('peak_price', 0.0)), now_price)
                peak_price = max(float(state.get('peak_price', 0.0)), 1e-8)
                peak_drawdown = (now_price - peak_price) / peak_price

                if profit <= -float(self.exit_rules['hard_stop_loss_pct']):
                    if self._submit_clear(stock, now_price, "【硬止损】触发清仓"):
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                    continue

                if (not state.get('tp2_done', False)) and profit >= float(self.exit_rules['tp2_pct']):
                    if self._submit_reduce(stock, now_price, "【第二止盈】执行减仓", ratio=self.exit_rules['tp2_sell_ratio']):
                        state['tp2_done'] = True
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                    continue

                if (not state.get('tp1_done', False)) and profit >= float(self.exit_rules['tp1_pct']):
                    if self._submit_reduce(stock, now_price, "【第一止盈】执行减仓", ratio=self.exit_rules['tp1_sell_ratio']):
                        state['tp1_done'] = True
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                    continue

                if profit > 0 and peak_drawdown <= -float(self.exit_rules['trail_clear_drawdown_pct']):
                    if self._submit_clear(stock, now_price, "【移动止盈】回撤过大清仓"):
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                    continue

                if (
                    (not state.get('trail_reduced', False)) and
                    profit >= float(self.exit_rules['trail_reduce_min_profit_pct']) and
                    peak_drawdown <= -float(self.exit_rules['trail_reduce_drawdown_pct'])
                ):
                    if self._submit_reduce(stock, now_price, "【移动止盈】回撤减仓", ratio=self.exit_rules['trail_reduce_sell_ratio']):
                        state['trail_reduced'] = True
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                    continue

                buy_sig = bool(
                    (cross_up or dif_up_turn) and
                    c_series.iloc[-1] < vwap and
                    near_lower and
                    c_series.iloc[-1] > c_series.iloc[-2] and
                    open_filter
                )
                sell_sig = bool(
                    (cross_down or dif_down_turn) and
                    c_series.iloc[-1] > vwap and
                    near_upper and
                    c_series.iloc[-1] < c_series.iloc[-2] and
                    open_filter
                )

                if urgent_pull and current_timestamp - state['last_alert'] >= 300:
                    if self._submit_signal(stock, SignalType.BUY_ADD, now_price, "↖急拉抢筹", default_qty=100):
                        state['last_alert'] = current_timestamp
                        state['last_buy_signal'] = current_timestamp
                        state['trade_count'] += 1
                elif urgent_drop and current_timestamp - state['last_alert'] >= 300:
                    if self._submit_signal(stock, SignalType.SELL_REDUCE, now_price, "↙急跌出逃", default_qty=100):
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                        state['trade_count'] += 1
                elif buy_sig and current_timestamp - state['last_buy_signal'] >= 600:
                    if self._submit_signal(stock, SignalType.BUY_ADD, now_price, "↖超跌低吸", default_qty=100):
                        state['last_alert'] = current_timestamp
                        state['last_buy_signal'] = current_timestamp
                        state['trade_count'] += 1
                elif sell_sig and current_timestamp - state['last_sell_signal'] >= 600:
                    if self._submit_signal(stock, SignalType.SELL_REDUCE, now_price, "↙轨道高抛", default_qty=100):
                        state['last_alert'] = current_timestamp
                        state['last_sell_signal'] = current_timestamp
                        state['trade_count'] += 1

            except Exception as e:
                pass

# =====================================================================
# 主执行入口：Pipeline 动态管线调度
# =====================================================================
def _load_trading_day_set(target_date: datetime) -> set:
    start_date = (target_date - timedelta(days=31)).strftime('%Y%m%d')
    end_date = (target_date + timedelta(days=31)).strftime('%Y%m%d')
    trading_dates = tq.get_trading_dates(market='SH', start_time=start_date, end_time=end_date, count=-1)
    return set(trading_dates or [])


def _classify_market_session(now: datetime, trading_day_set: set) -> str:
    today_str = now.strftime('%Y%m%d')
    if today_str not in trading_day_set:
        return 'closed_day'

    morning_open = datetime.strptime("09:30", "%H:%M").time()
    morning_close = datetime.strptime("11:30", "%H:%M").time()
    afternoon_open = datetime.strptime("13:00", "%H:%M").time()
    market_close = datetime.strptime("15:00", "%H:%M").time()

    current_time = now.time()
    if morning_open <= current_time <= morning_close:
        return 'trading'
    if afternoon_open <= current_time <= market_close:
        return 'trading'
    if morning_close < current_time < afternoon_open:
        return 'mid_break'
    return 'closed_hours'


def parse_args():
    parser = argparse.ArgumentParser(description="通达信智能量化系统：日线选股 + 盘中做T")
    parser.add_argument("--block-name", default="沪深300", help="策略显示用板块名称")
    parser.add_argument("--lookback-days", type=int, default=40, help="选股回看天数下限")
    parser.add_argument("--top-k", type=int, default=0, help="仅保留前 K 名，0 表示不限制数量")
    parser.add_argument("--target-tdx-block", default="威科夫共振", help="推送到通达信的自定义板块名称")
    parser.add_argument("--check-interval", type=int, default=10, help="盘中做T轮询间隔（秒）")
    parser.add_argument("--selection-interval", type=int, default=600, help="交易时段选股刷新间隔（秒）")
    parser.add_argument("--execution-mode", choices=["paper", "live"], default="paper", help="执行模式：paper 为模拟执行，live 为真实交易适配器骨架")
    parser.add_argument("--broker-config", default="", help="真实交易适配器配置文件路径（JSON）")
    parser.add_argument("--entry-position-ratio", type=float, default=0.10, help="首次建仓目标仓位，占总资产比例")
    parser.add_argument("--add-position-ratio", type=float, default=0.05, help="单次加仓目标仓位，占总资产比例")
    parser.add_argument("--min-trade-lot", type=int, default=100, help="最小交易手数，A 股通常为 100 股")
    parser.add_argument("--hard-stop-loss-pct", type=float, default=0.08, help="个股硬止损阈值，例如 0.08 表示亏损 8%% 清仓")
    parser.add_argument("--tp1-pct", type=float, default=0.10, help="第一止盈触发阈值")
    parser.add_argument("--tp1-sell-ratio", type=float, default=0.33, help="第一止盈减仓比例")
    parser.add_argument("--tp2-pct", type=float, default=0.18, help="第二止盈触发阈值")
    parser.add_argument("--tp2-sell-ratio", type=float, default=0.50, help="第二止盈减仓比例")
    parser.add_argument("--trail-reduce-drawdown-pct", type=float, default=0.06, help="移动止盈减仓的峰值回撤阈值")
    parser.add_argument("--trail-reduce-min-profit-pct", type=float, default=0.08, help="触发移动止盈减仓前要求的最低浮盈")
    parser.add_argument("--trail-reduce-sell-ratio", type=float, default=0.50, help="移动止盈减仓比例")
    parser.add_argument("--trail-clear-drawdown-pct", type=float, default=0.10, help="移动止盈清仓的峰值回撤阈值")
    parser.add_argument("--out-of-pool-clear-rounds", type=int, default=2, help="连续多少轮选股被调出后执行清仓")

    market_risk_group = parser.add_mutually_exclusive_group()
    market_risk_group.add_argument("--market-risk", dest="market_risk_enabled", action="store_true",
                                   help="开启大盘风控拦截")
    market_risk_group.add_argument("--no-market-risk", dest="market_risk_enabled", action="store_false",
                                   help="关闭大盘风控拦截")

    pool_exit_group = parser.add_mutually_exclusive_group()
    pool_exit_group.add_argument("--sell-if-out-of-pool", dest="sell_if_out_of_pool", action="store_true",
                                 help="选股刷新时，已持仓个股被调出股票池后触发减仓/清仓")
    pool_exit_group.add_argument("--hold-if-out-of-pool", dest="sell_if_out_of_pool", action="store_false",
                                 help="选股刷新时，即使个股被调出股票池仍继续持有并观察")
    parser.set_defaults(market_risk_enabled=False)
    parser.set_defaults(sell_if_out_of_pool=False)
    return parser.parse_args()


def build_strategy(args):
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    return AdvancedShortTermStrategyV6(
        block_name=args.block_name,
        lookback_days=args.lookback_days,
        top_k=top_k,
        target_tdx_block=args.target_tdx_block,
        market_risk_enabled=args.market_risk_enabled,
    )


def build_exit_rules(args):
    return {
        'hard_stop_loss_pct': args.hard_stop_loss_pct,
        'tp1_pct': args.tp1_pct,
        'tp1_sell_ratio': args.tp1_sell_ratio,
        'tp2_pct': args.tp2_pct,
        'tp2_sell_ratio': args.tp2_sell_ratio,
        'trail_reduce_drawdown_pct': args.trail_reduce_drawdown_pct,
        'trail_reduce_min_profit_pct': args.trail_reduce_min_profit_pct,
        'trail_reduce_sell_ratio': args.trail_reduce_sell_ratio,
        'trail_clear_drawdown_pct': args.trail_clear_drawdown_pct,
        'sell_if_out_of_pool': args.sell_if_out_of_pool,
        'out_of_pool_clear_rounds': args.out_of_pool_clear_rounds,
    }


def build_sizing_rules(args):
    return {
        'entry_position_ratio': args.entry_position_ratio,
        'add_position_ratio': args.add_position_ratio,
        'min_trade_lot': args.min_trade_lot,
    }


def main():
    args = parse_args()
    print("="*60)
    print(">>> 启动通达信智能量化系统：动态选股(10分钟) + 持续高频监控 <<<")
    print("="*60)
    
    tq.initialize(__file__)
    
    try:
        current_positions = {}
        execution_manager = build_execution_manager(args.execution_mode, args.broker_config or None)
        bot = IntradayRealtimeBotV2(
            positions=current_positions,
            execution_manager=execution_manager,
            exit_rules=build_exit_rules(args),
            sizing_rules=build_sizing_rules(args),
        )
        if args.execution_mode == "paper":
            print("[执行层] 已启用 PaperBroker 模拟执行，当前不会触发真实下单。")
        else:
            print("[执行层] 已切换到 live 适配器骨架。未实现真实券商接口前，任何下单动作都会抛出 NotImplementedError。")
        calendar_anchor = datetime.now().date()
        trading_day_set = _load_trading_day_set(datetime.now())
        
        check_interval = args.check_interval
        selection_interval = args.selection_interval
        last_selection_time = 0
        
        while True:
            now = datetime.now()
            if now.date() != calendar_anchor:
                calendar_anchor = now.date()
                trading_day_set = _load_trading_day_set(now)

            market_session = _classify_market_session(now, trading_day_set)

            # --- 非交易日或非交易时段模式：只运行一次选股并退出 ---
            if market_session in {'closed_day', 'closed_hours'}:
                if market_session == 'closed_day':
                    session_desc = "非交易日(周末/节假日)"
                else:
                    session_desc = "非交易时段(盘前/盘后)"
                print(f"\n[{now.strftime('%H:%M:%S')}] 系统检测到当前处于【{session_desc}】，进入静态选股模式。")
                print("===" * 15)
                strategy = build_strategy(args)
                success = strategy.run()
                if success and strategy.my_positions:
                    print(f"\n[+] 盘后静态选股完成！共选出 {len(strategy.my_positions)} 只标的，已推送至通达信自定义板块。")
                    print(f"    入选标的: {list(strategy.my_positions.keys())}")
                else:
                    print("\n[-] 盘后静态选股未找到符合共振条件的标的或被大盘风控拦截。")
                print("\n[V] 非交易时段任务执行完毕，系统即将退出。")
                break
                
            # --- 交易时段模式：动态管线 ---
            # 时间风控 (中午休市)
            if market_session == 'mid_break':
                time.sleep(60)
                continue
            
            current_timestamp = time.time()
            
            # 清理已触发清仓信号（止盈/止损）的股票
            cleared_stocks = [s for s in list(current_positions.keys()) if bot.trade_state.get(s, {}).get('cleared', False)]
            for s in cleared_stocks:
                print(f"\n      [-] 标的 {s} 已触发清仓信号，从动态监控池中永久移除。")
                del current_positions[s]
                bot.remove_stock(s)
            
            # --- 【阶段一】每 10 分钟运行一次选股模型 ---
            if current_timestamp - last_selection_time >= selection_interval:
                print(f"\n[{now.strftime('%H:%M:%S')}] === [定期研判] 执行威科夫+KAMA多维选股模型 ===")
                strategy = build_strategy(args)
                
                success = strategy.run()
                if success:
                    new_candidates = strategy.my_positions
                    latest_candidate_set = set(new_candidates.keys())
                    
                    # 增加新发现的标的，原有的坚决保留其原始成本价不动
                    for s, new_cost in new_candidates.items():
                        if s not in current_positions:
                            print(f"      [+] 新增潜力标的进入监控池: {s} (观察价: {new_cost:.2f}，等待抢筹建仓)")
                            current_positions[s] = None
                            bot.add_stock(s, None)
                            bot.trade_state[s]['selected_price'] = new_cost
                        else:
                            if current_positions[s] is None and s in bot.trade_state:
                                bot.trade_state[s]['selected_price'] = new_cost

                    bot.refresh_candidate_pool(latest_candidate_set)
                    print(f"      [!] 当前最新盘中动态监控池: {list(current_positions.keys())}")
                else:
                    print("      [!] 选股模型未产生有效信号，维持现有监控池。")
                    
                last_selection_time = time.time()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] === [底层通信] 恢复高速行情快照轮询 ===\n")
                
            # --- 【阶段二】毫秒级/秒级 高频快照盘口监控 ---
            bot.check_once()
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n[!] 用户手动中止了系统运行。")
    except Exception as e:
        print(f"\n[X] 系统运行出现异常: {e}")
        import traceback; traceback.print_exc()
    finally:
        tq.close()
        print("\n[V] 通达信 TQ 客户端连接已安全断开，进程退出。")

if __name__ == "__main__":
    main()
