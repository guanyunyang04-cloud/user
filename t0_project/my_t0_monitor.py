import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqcenter import tq
import warnings

warnings.filterwarnings('ignore')


MY_STOCKS_AND_COSTS = {
    '601678.SH': 5.50,
}


class ManualT0Monitor:
    def __init__(self, positions=None, exit_rules=None):
        self.my_positions = positions or {}
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
        }
        if exit_rules:
            self.exit_rules.update(exit_rules)
        self.trade_state = {
            stock: self._build_state(cost)
            for stock, cost in self.my_positions.items()
        }
        self.history_buffer = {stock: pd.DataFrame() for stock in self.my_positions}

    @staticmethod
    def _build_state(cost):
        init_price = float(cost) if cost is not None else 0.0
        return {
            'last_alert': 0.0,
            'last_buy_signal': 0.0,
            'last_sell_signal': 0.0,
            'selected_price': cost,
            'peak_price': init_price,
            'max_profit': 0.0,
            'tp1_alerted': False,
            'tp2_alerted': False,
            'trail_reduce_alerted': False,
            'trail_clear_alerted': False,
            'hard_stop_alerted': False,
        }

    def add_stock(self, stock, cost):
        self.my_positions[stock] = cost
        self.trade_state[stock] = self._build_state(cost)
        self.history_buffer[stock] = pd.DataFrame()

    def remove_stock(self, stock):
        self.my_positions.pop(stock, None)
        self.trade_state.pop(stock, None)
        self.history_buffer.pop(stock, None)

    def _send_alert(self, stock, price, bs_flag, reason, volume='100'):
        now_str = datetime.now().strftime('%Y%m%d%H%M%S')
        try:
            tq.send_warn(
                stock_list=[stock],
                time_list=[now_str],
                price_list=[str(price)],
                close_list=[str(price)],
                volum_list=[str(volume)],
                bs_flag_list=[str(bs_flag)],
                warn_type_list=['0'],
                reason_list=[reason],
                count=1,
            )
        except Exception:
            pass

        print(f'[{now_str[8:10]}:{now_str[10:12]}:{now_str[12:14]}] {stock} | {reason} | 现价: {price:.2f}')
        log_file = os.path.join(os.path.dirname(__file__), 't0_monitor_alerts.txt')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'[{now_str[8:10]}:{now_str[10:12]}:{now_str[12:14]}] {stock} | {reason} | 现价: {price:.2f} | 数量建议: {volume}\\n')
        except Exception:
            pass

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
            period='1m',
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
        if not self.my_positions:
            return

        for stock, cost in self.my_positions.items():
            state = self.trade_state[stock]
            try:
                snap = tq.get_market_snapshot(stock_code=stock)
                if not snap or snap.get('ErrorId') != '0':
                    continue

                now_price = float(snap.get('Now', 0))
                vwap = float(snap.get('Average', 0))
                if now_price <= 0 or vwap <= 0:
                    continue

                now_dt = datetime.now()
                minutes_passed = self._trading_minutes_passed(now_dt)
                open_filter = minutes_passed > 5

                df = self._load_intraday_bars(stock, count=120)
                if df.empty or len(df) < 15:
                    continue
                self.history_buffer[stock] = df

                c_series = df['Close']
                v_series = df['Volume']
                a_series = df['Amount']
                cum_volume = v_series.replace(0, np.nan).cumsum()
                vwap_series = a_series.cumsum() / cum_volume
                if (vwap_series > c_series * 50).any():
                    vwap_series = vwap_series / 100
                vwap_series = vwap_series.ffill().fillna(vwap)
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
                        2.0,
                    ),
                    index=df.index,
                    dtype=float,
                )
                dynamic_upper = vwap_series + multiplier * real_deviation
                dynamic_lower = vwap_series - multiplier * real_deviation
                transition_ratio = np.minimum(minute_bars / transition_period, 1.0)
                upper_band_series = initial_upper * (1.0 - transition_ratio) + dynamic_upper * transition_ratio
                lower_band_series = initial_lower * (1.0 - transition_ratio) + dynamic_lower * transition_ratio

                ema6 = c_series.ewm(span=6, adjust=False).mean()
                ema13 = c_series.ewm(span=13, adjust=False).mean()
                dif = ema6 - ema13
                dea = dif.ewm(span=5, adjust=False).mean()
                cross_up = self._cross_up_series(dif, dea)
                cross_down = self._cross_down_series(dif, dea)
                dif_up_turn = bool(dif.iloc[-1] > dif.iloc[-2] and dif.iloc[-2] <= dif.iloc[-3])
                dif_down_turn = bool(dif.iloc[-1] < dif.iloc[-2] and dif.iloc[-2] >= dif.iloc[-3])

                near_lower = bool((c_series.tail(5) <= lower_band_series.tail(5) * 1.001).any())
                near_upper = bool((c_series.tail(5) >= upper_band_series.tail(5) * 0.999).any())

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

                current_timestamp = time.time()
                ref_cost = cost if cost is not None else state.get('selected_price')

                if ref_cost is None or ref_cost <= 0:
                    if urgent_pull and current_timestamp - state['last_alert'] >= 300:
                        self._send_alert(stock, now_price, 0, '↖抢筹建仓', '100')
                        state['selected_price'] = now_price
                        state['peak_price'] = now_price
                        state['last_alert'] = current_timestamp
                        state['last_buy_signal'] = current_timestamp
                    continue

                profit = (now_price - ref_cost) / ref_cost
                state['max_profit'] = max(float(state.get('max_profit', 0.0)), profit)
                state['peak_price'] = max(float(state.get('peak_price', 0.0)), now_price)
                peak_price = max(float(state.get('peak_price', 0.0)), 1e-8)
                peak_drawdown = (now_price - peak_price) / peak_price

                if (not state['hard_stop_alerted']) and profit <= -float(self.exit_rules['hard_stop_loss_pct']):
                    self._send_alert(stock, now_price, 1, '【硬止损】触发清仓', '全仓')
                    state['hard_stop_alerted'] = True
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                    continue

                if (not state['tp2_alerted']) and profit >= float(self.exit_rules['tp2_pct']):
                    self._send_alert(stock, now_price, 1, '【第二止盈】执行减仓', f"{int(self.exit_rules['tp2_sell_ratio'] * 100)}%仓位")
                    state['tp2_alerted'] = True
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                    continue

                if (not state['tp1_alerted']) and profit >= float(self.exit_rules['tp1_pct']):
                    self._send_alert(stock, now_price, 1, '【第一止盈】执行减仓', f"{int(self.exit_rules['tp1_sell_ratio'] * 100)}%仓位")
                    state['tp1_alerted'] = True
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                    continue

                if (not state['trail_clear_alerted']) and profit > 0 and peak_drawdown <= -float(self.exit_rules['trail_clear_drawdown_pct']):
                    self._send_alert(stock, now_price, 1, '【移动止盈】回撤过大清仓', '全仓')
                    state['trail_clear_alerted'] = True
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                    continue

                if (
                    (not state['trail_reduce_alerted']) and
                    profit >= float(self.exit_rules['trail_reduce_min_profit_pct']) and
                    peak_drawdown <= -float(self.exit_rules['trail_reduce_drawdown_pct'])
                ):
                    self._send_alert(stock, now_price, 1, '【移动止盈】回撤减仓', f"{int(self.exit_rules['trail_reduce_sell_ratio'] * 100)}%仓位")
                    state['trail_reduce_alerted'] = True
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                    continue

                if urgent_pull and current_timestamp - state['last_alert'] >= 300:
                    self._send_alert(stock, now_price, 0, '↖急拉抢筹', '加仓')
                    state['last_alert'] = current_timestamp
                    state['last_buy_signal'] = current_timestamp
                elif urgent_drop and current_timestamp - state['last_alert'] >= 300:
                    self._send_alert(stock, now_price, 1, '↙急跌出逃', '减仓')
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
                elif buy_sig and current_timestamp - state['last_buy_signal'] >= 600:
                    self._send_alert(stock, now_price, 0, '↖超跌低吸', '加仓')
                    state['last_alert'] = current_timestamp
                    state['last_buy_signal'] = current_timestamp
                elif sell_sig and current_timestamp - state['last_sell_signal'] >= 600:
                    self._send_alert(stock, now_price, 1, '↙轨道高抛', '减仓')
                    state['last_alert'] = current_timestamp
                    state['last_sell_signal'] = current_timestamp
            except Exception:
                pass


def _load_trading_day_set(target_date: datetime) -> set:
    start_date = (target_date - timedelta(days=31)).strftime('%Y%m%d')
    end_date = (target_date + timedelta(days=31)).strftime('%Y%m%d')
    trading_dates = tq.get_trading_dates(market='SH', start_time=start_date, end_time=end_date, count=-1)
    return set(trading_dates or [])


def _classify_market_session(now: datetime, trading_day_set: set) -> str:
    today_str = now.strftime('%Y%m%d')
    if today_str not in trading_day_set:
        return 'closed_day'

    morning_open = datetime.strptime('09:30', '%H:%M').time()
    morning_close = datetime.strptime('11:30', '%H:%M').time()
    afternoon_open = datetime.strptime('13:00', '%H:%M').time()
    market_close = datetime.strptime('15:00', '%H:%M').time()

    current_time = now.time()
    if morning_open <= current_time <= morning_close:
        return 'trading'
    if afternoon_open <= current_time <= market_close:
        return 'trading'
    if morning_close < current_time < afternoon_open:
        return 'mid_break'
    return 'closed_hours'


def parse_args():
    parser = argparse.ArgumentParser(description='手工做T监控器：当前盘中策略同步版')
    parser.add_argument('--positions', default='600097.SH:12.37', help='传入监控标的，格式示例：601678.SH:5.50,600036.SH:36.20,601318.SH:none')
    parser.add_argument('--check-interval', type=int, default=10, help='交易时段轮询间隔，单位秒')
    parser.add_argument('--hard-stop-loss-pct', type=float, default=0.08, help='个股硬止损阈值，例如 0.08 表示亏损 8%% 清仓提醒')
    parser.add_argument('--tp1-pct', type=float, default=0.10, help='第一止盈触发阈值')
    parser.add_argument('--tp1-sell-ratio', type=float, default=0.33, help='第一止盈减仓比例')
    parser.add_argument('--tp2-pct', type=float, default=0.18, help='第二止盈触发阈值')
    parser.add_argument('--tp2-sell-ratio', type=float, default=0.50, help='第二止盈减仓比例')
    parser.add_argument('--trail-reduce-drawdown-pct', type=float, default=0.06, help='移动止盈减仓的峰值回撤阈值')
    parser.add_argument('--trail-reduce-min-profit-pct', type=float, default=0.08, help='触发移动止盈减仓前要求的最低浮盈')
    parser.add_argument('--trail-reduce-sell-ratio', type=float, default=0.50, help='移动止盈减仓比例')
    parser.add_argument('--trail-clear-drawdown-pct', type=float, default=0.10, help='移动止盈清仓的峰值回撤阈值')
    return parser.parse_args()


def parse_positions_arg(raw_text):
    if not raw_text:
        return dict(MY_STOCKS_AND_COSTS)

    positions = {}
    for item in raw_text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'无效持仓格式: {item}')
        stock, cost_text = item.split(':', 1)
        stock = stock.strip()
        cost_text = cost_text.strip().lower()
        if not stock:
            raise ValueError(f'无效股票代码: {item}')
        if cost_text in {'none', 'null', ''}:
            positions[stock] = None
        else:
            positions[stock] = float(cost_text)
    return positions


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
    }


def main():
    args = parse_args()
    print('=' * 58)
    print('>>> 启动手工做T监控器：当前盘中策略同步版 <<<')
    print('=' * 58)

    try:
        positions = parse_positions_arg(args.positions)
    except ValueError as exc:
        print(f'[X] 参数解析失败: {exc}')
        return

    if not positions:
        print('[!] 监控标的为空，程序退出。')
        return

    tq.initialize(__file__)
    try:
        bot = ManualT0Monitor(positions=dict(positions), exit_rules=build_exit_rules(args))
        trading_day_set = _load_trading_day_set(datetime.now())
        calendar_anchor = datetime.now().date()

        for stock, cost in positions.items():
            if cost is None:
                print(f'[+] 加入观察标的: {stock} (等待抢筹建仓信号)')
            else:
                print(f'[+] 加入持仓监控: {stock} | 成本价: {cost:.2f}')

        print(f'[V] 监控器初始化完成，交易时段每 {args.check_interval} 秒轮询一次。')

        while True:
            now = datetime.now()
            if now.date() != calendar_anchor:
                calendar_anchor = now.date()
                trading_day_set = _load_trading_day_set(now)

            market_session = _classify_market_session(now, trading_day_set)
            if market_session == 'closed_day':
                print(f'[{now.strftime("%H:%M:%S")}] 非交易日，监控器退出。')
                break
            if market_session == 'closed_hours':
                print(f'[{now.strftime("%H:%M:%S")}] 非交易时段，监控器退出。')
                break
            if market_session == 'mid_break':
                time.sleep(60)
                continue

            bot.check_once()
            time.sleep(args.check_interval)
    except KeyboardInterrupt:
        print('\\n[!] 用户手动中止监控。')
    finally:
        tq.close()
        print('[V] TQ 连接已关闭。')


if __name__ == '__main__':
    main()
