import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tqcenter import tq
from integrated_tq_strategy import AdvancedShortTermStrategyV6, tqdm


ZH_METRIC_LABELS = {
    "initial_capital": "初始本金",
    "ending_equity": "期末总资产",
    "total_return": "累计收益率",
    "annual_return": "年化收益率",
    "annual_vol": "年化波动率",
    "sharpe": "夏普比率",
    "max_drawdown": "最大回撤",
    "win_rate": "日度胜率",
    "avg_holding_count": "平均持仓数",
    "holding_day_ratio": "非空仓天数占比",
    "selection_hit_ratio": "有选股结果的更新日占比",
    "cash_ratio_end": "期末现金占比",
    "rebalance_count": "股票池更新次数",
    "hard_stop_count": "硬止损清仓次数",
    "take_profit_reduce_count": "止盈减仓次数",
    "trailing_reduce_count": "移动止盈减仓次数",
    "trailing_clear_count": "移动止盈清仓次数",
    "pool_reduce_count": "调出池减仓次数",
    "pool_clear_count": "调出池清仓次数",
    "avg_forward_5d": "入选后5日平均收益",
    "avg_forward_10d": "入选后10日平均收益",
    "avg_forward_20d": "入选后20日平均收益",
    "universe_size": "股票池数量",
}

PERCENT_KEYS = {
    "total_return",
    "annual_return",
    "annual_vol",
    "max_drawdown",
    "win_rate",
    "holding_day_ratio",
    "selection_hit_ratio",
    "cash_ratio_end",
    "avg_forward_5d",
    "avg_forward_10d",
    "avg_forward_20d",
}


def parse_args():
    parser = argparse.ArgumentParser(description="集成日线选股策略的离线回测与评估框架")
    parser.add_argument("--stocks", default="", help="逗号分隔股票列表；为空时默认全市场主板")
    parser.add_argument("--start-date", default="20250315", help="回测开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", default="", help="回测结束日期，格式 YYYYMMDD，默认今天")
    parser.add_argument("--top-k", type=int, default=20, help="每日股票池最多保留前 K 名，0 表示不限制")
    parser.add_argument("--max-positions", type=int, default=10, help="组合最多持有几只股票")
    parser.add_argument("--pool-refresh-days", type=int, default=1, help="股票池更新频率，单位交易日，默认每日更新")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="单边交易成本，单位 bps")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0, help="初始本金")
    parser.add_argument("--lot-size", type=int, default=100, help="最小交易单位，A股默认 100 股")
    parser.add_argument("--hard-stop-loss-pct", type=float, default=0.08, help="硬止损阈值，例如 0.08 表示亏损 8%% 清仓")
    parser.add_argument("--tp1-pct", type=float, default=0.10, help="第一止盈阈值，例如 0.10 表示盈利 10%%")
    parser.add_argument("--tp1-sell-ratio", type=float, default=0.34, help="第一止盈减仓比例，例如 0.34 表示卖出约三分之一")
    parser.add_argument("--tp2-pct", type=float, default=0.18, help="第二止盈阈值，例如 0.18 表示盈利 18%%")
    parser.add_argument("--tp2-sell-ratio", type=float, default=0.50, help="第二止盈减仓比例，例如 0.50 表示卖出当前剩余仓位的一半")
    parser.add_argument("--trail-reduce-drawdown-pct", type=float, default=0.06, help="从持仓后最高收盘价回撤达到该比例时触发移动止盈减仓")
    parser.add_argument("--trail-reduce-min-profit-pct", type=float, default=0.08, help="触发移动止盈减仓前，当前盈利至少达到该比例")
    parser.add_argument("--trail-reduce-sell-ratio", type=float, default=0.50, help="移动止盈减仓比例")
    parser.add_argument("--trail-clear-drawdown-pct", type=float, default=0.10, help="从持仓后最高收盘价回撤达到该比例时触发移动止盈清仓")
    parser.add_argument("--sell-if-out-of-pool", action="store_true", help="持仓股被调出最新股票池时启用减仓/清仓流程")
    parser.add_argument("--out-of-pool-clear-days", type=int, default=2, help="连续多少天不在股票池中则清仓，默认 2 天")
    parser.add_argument("--output-dir", default="", help="结果输出目录，默认写入 backtest_output/时间戳")

    market_risk_group = parser.add_mutually_exclusive_group()
    market_risk_group.add_argument("--market-risk", dest="market_risk_enabled", action="store_true", help="开启大盘风控过滤")
    market_risk_group.add_argument("--no-market-risk", dest="market_risk_enabled", action="store_false", help="关闭大盘风控过滤")
    parser.set_defaults(market_risk_enabled=False)
    return parser.parse_args()


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)
    return normalized.sort_index()


def load_universe(stocks_arg: str) -> list[str]:
    if stocks_arg.strip():
        return [s.strip() for s in stocks_arg.split(",") if s.strip()]
    all_stocks = tq.get_stock_list()
    return [s for s in all_stocks if s.startswith("60") or s.startswith("00")]


def fetch_daily_data(stocks: list[str], start_date: str, end_date: str, preload_days: int = 800) -> dict[str, pd.DataFrame]:
    end_dt = datetime.strptime(end_date, "%Y%m%d") if end_date else datetime.now()
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    fetch_start = (start_dt - timedelta(days=preload_days)).strftime("%Y%m%d")
    fetch_end = end_dt.strftime("%Y%m%d")
    fetch_count = max((end_dt - start_dt).days + preload_days, 600)
    stock_list = list(dict.fromkeys(stocks + ["999999.SH"]))
    df_dict = tq.get_market_data(
        field_list=["Close", "Open", "High", "Low", "Volume", "Amount"],
        stock_list=stock_list,
        start_time=fetch_start,
        end_time=fetch_end,
        count=fetch_count,
        dividend_type="front",
        period="1d",
    )
    return {field: normalize_index(frame) for field, frame in df_dict.items()}


def build_trade_dates(close_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else close_df.index.max()
    mask = (close_df.index >= start_dt) & (close_df.index <= end_dt)
    return close_df.index[mask]


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def floor_to_lot(quantity: int, lot_size: int) -> int:
    quantity = int(quantity)
    return max((quantity // lot_size) * lot_size, 0)


def forward_return(close_df: pd.DataFrame, stock: str, signal_dt: pd.Timestamp, horizon: int) -> float | None:
    if stock not in close_df.columns:
        return None
    date_locs = close_df.index.get_indexer([signal_dt])
    if len(date_locs) == 0 or date_locs[0] < 0:
        return None
    start_loc = int(date_locs[0])
    end_loc = start_loc + horizon
    if end_loc >= len(close_df.index):
        return None
    start_px = close_df.iloc[start_loc][stock]
    end_px = close_df.iloc[end_loc][stock]
    if pd.isna(start_px) or pd.isna(end_px) or start_px == 0:
        return None
    return float(end_px / start_px - 1.0)


def build_daily_pool_schedule(
    df_dict: dict[str, pd.DataFrame],
    strategy: AdvancedShortTermStrategyV6,
    trade_dates: pd.DatetimeIndex,
    pool_refresh_days: int,
    max_positions: int,
) -> tuple[dict[pd.Timestamp, dict], pd.DataFrame, pd.DataFrame]:
    close_df = df_dict["Close"].copy()
    universe = [col for col in close_df.columns if col != "999999.SH"]
    pool_schedule = {}
    rebalance_logs = []
    selection_logs = []

    signal_indices = list(range(0, max(len(trade_dates) - 1, 0), max(pool_refresh_days, 1)))
    date_iter = signal_indices
    if tqdm is not None:
        date_iter = tqdm(signal_indices, desc="生成每日股票池", unit="次", dynamic_ncols=True, ascii=True)

    for signal_idx in date_iter:
        signal_dt = trade_dates[signal_idx]
        entry_idx = signal_idx + 1
        if entry_idx >= len(trade_dates):
            break

        sliced = {field: frame.loc[:signal_dt] for field, frame in df_dict.items()}
        ranked = strategy.select_from_history(sliced, execute_signals=False)
        selected = [s for s in ranked.index.tolist() if s in universe][:max_positions]
        next_open_dt = trade_dates[entry_idx]
        pool_schedule[next_open_dt] = {
            "signal_date": signal_dt,
            "targets": selected,
            "ranked": ranked.loc[selected].copy() if selected else ranked.iloc[0:0].copy(),
        }

        rebalance_logs.append({
            "信号日期": signal_dt.strftime("%Y-%m-%d"),
            "次日开盘执行日": next_open_dt.strftime("%Y-%m-%d"),
            "入选数量": len(selected),
            "入选股票": ",".join(selected),
        })

        top_slice = ranked[["Total_Score", "Signal_Tag"]].copy()
        for stock, row in top_slice.iterrows():
            selection_logs.append({
                "信号日期": signal_dt.strftime("%Y-%m-%d"),
                "股票代码": stock,
                "评分": float(row["Total_Score"]),
                "信号标签": row["Signal_Tag"],
                "后5日收益": forward_return(close_df, stock, signal_dt, 5),
                "后10日收益": forward_return(close_df, stock, signal_dt, 10),
                "后20日收益": forward_return(close_df, stock, signal_dt, 20),
            })

    return pool_schedule, pd.DataFrame(rebalance_logs), pd.DataFrame(selection_logs)


def make_position(shares: int, price: float, dt: pd.Timestamp) -> dict:
    return {
        "shares": int(shares),
        "cost": float(price),
        "entry_date": dt.strftime("%Y-%m-%d"),
        "peak_close": float(price),
        "tp1_done": False,
        "tp2_done": False,
        "trail_reduced": False,
        "pool_reduced": False,
        "out_of_pool_days": 0,
    }


def execute_open_buys(
    dt: pd.Timestamp,
    targets: list[str],
    positions: dict[str, dict],
    cash: float,
    open_prices: pd.Series,
    cost_rate: float,
    lot_size: int,
    max_positions: int,
) -> tuple[dict[str, dict], float, list[dict]]:
    trades = []
    current_codes = set(positions.keys())
    candidates = [s for s in targets if s not in current_codes]
    slots = max(max_positions - len(current_codes), 0)
    buy_list = candidates[:slots]

    if not buy_list or cash <= 0:
        return positions, cash, trades

    remaining_slots = len(buy_list)
    for stock in buy_list:
        px = open_prices.get(stock, np.nan)
        if pd.isna(px) or px <= 0:
            remaining_slots -= 1
            continue

        budget = cash / max(remaining_slots, 1)
        shares = floor_to_lot(int(budget / (px * (1.0 + cost_rate))), lot_size)
        if shares <= 0:
            remaining_slots -= 1
            continue

        gross_amount = shares * px
        fee = gross_amount * cost_rate
        total_cost = gross_amount + fee
        if total_cost > cash:
            shares = floor_to_lot(int(cash / (px * (1.0 + cost_rate))), lot_size)
            gross_amount = shares * px
            fee = gross_amount * cost_rate
            total_cost = gross_amount + fee
        if shares <= 0 or total_cost > cash:
            remaining_slots -= 1
            continue

        cash -= total_cost
        positions[stock] = make_position(shares, float(px), dt)
        trades.append({
            "日期": dt.strftime("%Y-%m-%d"),
            "时点": "开盘",
            "股票代码": stock,
            "方向": "买入",
            "价格": float(px),
            "数量": int(shares),
            "成交金额": float(gross_amount),
            "交易成本": float(fee),
            "原因": "根据前一日更新后的股票池在开盘建仓",
        })
        remaining_slots -= 1

    return positions, cash, trades


def sell_position(
    dt: pd.Timestamp,
    stock: str,
    pos: dict,
    price: float,
    ratio: float,
    cost_rate: float,
    lot_size: int,
    reason: str,
    clear_all: bool = False,
) -> tuple[dict | None, float, dict | None]:
    current_shares = int(pos["shares"])
    if current_shares <= 0 or pd.isna(price) or price <= 0:
        return pos, 0.0, None

    if clear_all:
        sell_shares = current_shares
    else:
        sell_shares = floor_to_lot(int(current_shares * ratio), lot_size)
        if sell_shares <= 0 and current_shares >= lot_size:
            sell_shares = lot_size
        sell_shares = min(sell_shares, current_shares)

    if sell_shares <= 0:
        return pos, 0.0, None

    gross_amount = sell_shares * price
    fee = gross_amount * cost_rate
    cash_delta = gross_amount - fee
    remaining = current_shares - sell_shares
    trade = {
        "日期": dt.strftime("%Y-%m-%d"),
        "时点": "尾盘",
        "股票代码": stock,
        "方向": "卖出",
        "价格": float(price),
        "数量": int(sell_shares),
        "成交金额": float(gross_amount),
        "交易成本": float(fee),
        "原因": reason,
    }

    if remaining <= 0:
        return None, cash_delta, trade

    pos = dict(pos)
    pos["shares"] = remaining
    return pos, cash_delta, trade


def execute_close_actions(
    dt: pd.Timestamp,
    targets_today: set[str],
    positions: dict[str, dict],
    close_prices: pd.Series,
    cost_rate: float,
    lot_size: int,
    args,
) -> tuple[dict[str, dict], float, list[dict], dict[str, int]]:
    cash_delta = 0.0
    trades = []
    stats = {
        "hard_stop_count": 0,
        "take_profit_reduce_count": 0,
        "trailing_reduce_count": 0,
        "trailing_clear_count": 0,
        "pool_reduce_count": 0,
        "pool_clear_count": 0,
    }

    for stock in list(positions.keys()):
        pos = positions.get(stock)
        if pos is None:
            continue
        close_px = close_prices.get(stock, np.nan)
        if pd.isna(close_px) or close_px <= 0:
            continue

        pos["peak_close"] = max(float(pos.get("peak_close", close_px)), float(close_px))
        pnl_ratio = close_px / max(pos["cost"], 1e-8) - 1.0
        peak_drawdown = close_px / max(pos["peak_close"], 1e-8) - 1.0

        if stock not in targets_today:
            pos["out_of_pool_days"] = int(pos.get("out_of_pool_days", 0)) + 1
        else:
            pos["out_of_pool_days"] = 0
            pos["pool_reduced"] = False

        action = None
        stat_key = None
        ratio = 0.0
        clear_all = False

        if pnl_ratio <= -float(args.hard_stop_loss_pct):
            action = "触发硬止损，尾盘清仓"
            stat_key = "hard_stop_count"
            clear_all = True
        elif not pos.get("tp2_done", False) and pnl_ratio >= float(args.tp2_pct):
            action = "达到第二止盈阈值，尾盘减仓"
            stat_key = "take_profit_reduce_count"
            ratio = float(args.tp2_sell_ratio)
        elif not pos.get("tp1_done", False) and pnl_ratio >= float(args.tp1_pct):
            action = "达到第一止盈阈值，尾盘减仓"
            stat_key = "take_profit_reduce_count"
            ratio = float(args.tp1_sell_ratio)
        elif peak_drawdown <= -float(args.trail_clear_drawdown_pct) and pnl_ratio > 0:
            action = "触发移动止盈清仓，尾盘卖出"
            stat_key = "trailing_clear_count"
            clear_all = True
        elif (
            not pos.get("trail_reduced", False)
            and peak_drawdown <= -float(args.trail_reduce_drawdown_pct)
            and pnl_ratio >= float(args.trail_reduce_min_profit_pct)
        ):
            action = "触发移动止盈减仓，尾盘卖出部分仓位"
            stat_key = "trailing_reduce_count"
            ratio = float(args.trail_reduce_sell_ratio)
        elif bool(args.sell_if_out_of_pool) and pos.get("out_of_pool_days", 0) >= int(args.out_of_pool_clear_days):
            action = "连续多日调出股票池，尾盘清仓"
            stat_key = "pool_clear_count"
            clear_all = True
        elif bool(args.sell_if_out_of_pool) and pos.get("out_of_pool_days", 0) >= 1 and not pos.get("pool_reduced", False):
            action = "调出股票池，先尾盘减仓"
            stat_key = "pool_reduce_count"
            ratio = 0.50

        if action is None:
            positions[stock] = pos
            continue

        new_pos, delta_cash, trade = sell_position(
            dt=dt,
            stock=stock,
            pos=pos,
            price=float(close_px),
            ratio=ratio,
            cost_rate=cost_rate,
            lot_size=lot_size,
            reason=action,
            clear_all=clear_all,
        )
        if trade is None:
            positions[stock] = pos
            continue

        cash_delta += delta_cash
        trades.append(trade)
        stats[stat_key] += 1

        if stat_key == "take_profit_reduce_count":
            if pnl_ratio >= float(args.tp2_pct):
                pos["tp2_done"] = True
            else:
                pos["tp1_done"] = True
        elif stat_key == "trailing_reduce_count":
            pos["trail_reduced"] = True
        elif stat_key == "pool_reduce_count":
            pos["pool_reduced"] = True

        if new_pos is None:
            positions.pop(stock, None)
        else:
            merged = dict(pos)
            merged["shares"] = new_pos["shares"]
            positions[stock] = merged

    return positions, cash_delta, trades, stats


def run_backtest(
    df_dict: dict[str, pd.DataFrame],
    strategy: AdvancedShortTermStrategyV6,
    trade_dates: pd.DatetimeIndex,
    cost_rate: float,
    initial_capital: float,
    max_positions: int,
    lot_size: int,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    close_df = df_dict["Close"].copy()
    open_df = df_dict["Open"].copy()
    pool_schedule, rebalances_df, selections_df = build_daily_pool_schedule(
        df_dict=df_dict,
        strategy=strategy,
        trade_dates=trade_dates,
        pool_refresh_days=max(args.pool_refresh_days, 1),
        max_positions=max_positions,
    )

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    current_targets: set[str] = set()
    equity_rows = []
    portfolio_rows = []
    trade_logs = []
    aggregate_stats = {
        "hard_stop_count": 0,
        "take_profit_reduce_count": 0,
        "trailing_reduce_count": 0,
        "trailing_clear_count": 0,
        "pool_reduce_count": 0,
        "pool_clear_count": 0,
    }

    date_iter = trade_dates
    if tqdm is not None:
        date_iter = tqdm(trade_dates, desc="执行日频回测", unit="日", dynamic_ncols=True, ascii=True)

    for dt in date_iter:
        open_prices = open_df.loc[dt]
        close_prices = close_df.loc[dt]
        buy_count = 0
        sell_count = 0

        if dt in pool_schedule:
            current_targets = set(pool_schedule[dt]["targets"])
            positions, cash, open_trades = execute_open_buys(
                dt=dt,
                targets=pool_schedule[dt]["targets"],
                positions=positions,
                cash=cash,
                open_prices=open_prices,
                cost_rate=cost_rate,
                lot_size=lot_size,
                max_positions=max_positions,
            )
            trade_logs.extend(open_trades)
            buy_count += len(open_trades)

        positions, cash_sell, close_trades, close_stats = execute_close_actions(
            dt=dt,
            targets_today=current_targets,
            positions=positions,
            close_prices=close_prices,
            cost_rate=cost_rate,
            lot_size=lot_size,
            args=args,
        )
        cash += cash_sell
        trade_logs.extend(close_trades)
        sell_count += len(close_trades)
        for key in aggregate_stats:
            aggregate_stats[key] += close_stats[key]

        market_value = 0.0
        holding_count = 0
        for stock, pos in positions.items():
            px = close_prices.get(stock, np.nan)
            if pd.isna(px) or px <= 0:
                px = open_prices.get(stock, np.nan)
            if pd.isna(px) or px <= 0:
                continue
            market_value += pos["shares"] * px
            holding_count += 1

        equity = cash + market_value
        cash_ratio = float(cash / equity) if equity > 0 else 1.0
        equity_rows.append({"date": dt, "equity": equity})
        portfolio_rows.append({
            "date": dt,
            "cash": cash,
            "market_value": market_value,
            "holding_count": holding_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "cash_ratio": cash_ratio,
        })

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    portfolio_df = pd.DataFrame(portfolio_rows).set_index("date")
    trades_df = pd.DataFrame(trade_logs)
    return equity_df, portfolio_df, rebalances_df, selections_df, trades_df, aggregate_stats


def summarize_selection_outcomes(selections_df: pd.DataFrame) -> dict[str, float]:
    if selections_df.empty:
        return {"avg_forward_5d": 0.0, "avg_forward_10d": 0.0, "avg_forward_20d": 0.0}
    return {
        "avg_forward_5d": float(selections_df["后5日收益"].dropna().mean()) if "后5日收益" in selections_df else 0.0,
        "avg_forward_10d": float(selections_df["后10日收益"].dropna().mean()) if "后10日收益" in selections_df else 0.0,
        "avg_forward_20d": float(selections_df["后20日收益"].dropna().mean()) if "后20日收益" in selections_df else 0.0,
    }


def compute_metrics(
    equity_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    rebalances_df: pd.DataFrame,
    initial_capital: float,
    aggregate_stats: dict[str, int],
) -> dict[str, float]:
    daily_ret = equity_df["equity"].pct_change().dropna()
    total_return = float(equity_df["equity"].iloc[-1] / initial_capital - 1.0)
    annual_return = float((equity_df["equity"].iloc[-1] / initial_capital) ** (252 / max(len(daily_ret), 1)) - 1.0) if len(equity_df) > 1 else 0.0
    annual_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else 0.0
    win_rate = float((daily_ret > 0).mean()) if not daily_ret.empty else 0.0
    avg_holdings = float(portfolio_df["holding_count"].mean()) if not portfolio_df.empty else 0.0
    holding_day_ratio = float((portfolio_df["holding_count"] > 0).mean()) if not portfolio_df.empty else 0.0
    selection_hit_ratio = float((rebalances_df["入选数量"] > 0).mean()) if not rebalances_df.empty else 0.0
    cash_ratio_end = float(portfolio_df["cash_ratio"].iloc[-1]) if not portfolio_df.empty else 1.0

    metrics = {
        "initial_capital": float(initial_capital),
        "ending_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity_df["equity"]),
        "win_rate": win_rate,
        "avg_holding_count": avg_holdings,
        "holding_day_ratio": holding_day_ratio,
        "selection_hit_ratio": selection_hit_ratio,
        "cash_ratio_end": cash_ratio_end,
        "rebalance_count": int(len(rebalances_df)),
    }
    metrics.update({k: int(v) for k, v in aggregate_stats.items()})
    return metrics


def format_metric_value(key: str, value) -> str:
    if key in {"initial_capital", "ending_equity"}:
        return f"{value:,.2f} 元"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if key in PERCENT_KEYS:
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_chinese_metrics(metrics: dict[str, float]) -> dict[str, str]:
    return {ZH_METRIC_LABELS.get(key, key): format_metric_value(key, value) for key, value in metrics.items()}


def build_interpretation(metrics: dict[str, float]) -> list[str]:
    notes = []
    if metrics.get("holding_day_ratio", 0.0) < 0.1:
        notes.append("大部分时间空仓或轻仓，说明选股条件较严或卖出管理较积极。")
    elif metrics.get("holding_day_ratio", 0.0) < 0.3:
        notes.append("持仓参与率偏低，策略更像机会型出手。")
    else:
        notes.append("策略有一定的持续持仓能力。")

    if metrics.get("avg_forward_20d", 0.0) > 0.05:
        notes.append("入选股票后续20日平均表现较强，说明选股方向可能有效。")
    else:
        notes.append("入选股票后续表现一般，选股信号仍需继续优化。")

    if metrics.get("hard_stop_count", 0) > (metrics.get("take_profit_reduce_count", 0) + metrics.get("trailing_reduce_count", 0)):
        notes.append("硬止损次数偏多，需要重点检查建仓时机和风险阈值。")
    else:
        notes.append("减仓与清仓节奏相对更偏向顺势管理。")

    if metrics.get("max_drawdown", 0.0) < -0.2:
        notes.append("最大回撤偏大，说明现有减仓清仓体系仍需加强。")
    elif metrics.get("max_drawdown", 0.0) < -0.1:
        notes.append("最大回撤中等，需要结合收益看是否值得。")
    else:
        notes.append("最大回撤控制得相对温和。")
    return notes


def build_summary_text(args, metrics: dict[str, float], chinese_metrics: dict[str, str]) -> str:
    lines = [
        "回测结果摘要",
        f"回测区间：{args.start_date} - {args.end_date or datetime.now().strftime('%Y%m%d')}",
        f"股票池数量：{metrics.get('universe_size', 0)}",
        f"股票池更新频率：每 {args.pool_refresh_days} 个交易日",
        f"最大持仓数：{args.max_positions}",
        f"初始本金：{format_metric_value('initial_capital', metrics.get('initial_capital', 0.0))}",
        f"硬止损阈值：{args.hard_stop_loss_pct * 100:.2f}%",
        f"第一止盈阈值：{args.tp1_pct * 100:.2f}%",
        f"第二止盈阈值：{args.tp2_pct * 100:.2f}%",
        f"交易成本：单边 {args.cost_bps:.2f} bps",
        "",
        "核心指标：",
    ]
    for key in [
        "ending_equity",
        "total_return",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "holding_day_ratio",
        "selection_hit_ratio",
        "hard_stop_count",
        "take_profit_reduce_count",
        "trailing_reduce_count",
        "trailing_clear_count",
        "avg_forward_20d",
    ]:
        label = ZH_METRIC_LABELS[key]
        lines.append(f"- {label}：{chinese_metrics[label]}")

    lines.append("")
    lines.append("简要解读：")
    for note in build_interpretation(metrics):
        lines.append(f"- {note}")
    return "\n".join(lines)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "backtest_output" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tq.initialize(__file__)
    try:
        stocks = load_universe(args.stocks)
        strategy = AdvancedShortTermStrategyV6(
            lookback_days=520,
            top_k=args.top_k if args.top_k > 0 else None,
            market_risk_enabled=args.market_risk_enabled,
            enable_logs=False,
        )
        df_dict = fetch_daily_data(stocks, args.start_date, args.end_date)
        trade_dates = build_trade_dates(df_dict["Close"], args.start_date, args.end_date)
        if len(trade_dates) < 3:
            raise ValueError("有效交易日不足，无法完成回测。")

        equity_df, portfolio_df, rebalances_df, selections_df, trades_df, aggregate_stats = run_backtest(
            df_dict=df_dict,
            strategy=strategy,
            trade_dates=trade_dates,
            cost_rate=args.cost_bps / 10000.0,
            initial_capital=float(args.initial_capital),
            max_positions=max(int(args.max_positions), 1),
            lot_size=max(int(args.lot_size), 1),
            args=args,
        )

        metrics = compute_metrics(
            equity_df=equity_df,
            portfolio_df=portfolio_df,
            rebalances_df=rebalances_df,
            initial_capital=float(args.initial_capital),
            aggregate_stats=aggregate_stats,
        )
        metrics.update(summarize_selection_outcomes(selections_df))
        metrics["universe_size"] = len(stocks)

        chinese_metrics = build_chinese_metrics(metrics)
        summary_text = build_summary_text(args, metrics, chinese_metrics)

        equity_df.to_csv(output_dir / "equity_curve.csv")
        portfolio_df.to_csv(output_dir / "portfolio_daily.csv")
        rebalances_df.to_csv(output_dir / "rebalances.csv", index=False)
        selections_df.to_csv(output_dir / "selections.csv", index=False)
        trades_df.to_csv(output_dir / "trades.csv", index=False)

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(output_dir / "metrics_zh.json", "w", encoding="utf-8") as f:
            json.dump(chinese_metrics, f, ensure_ascii=False, indent=2)
        with open(output_dir / "summary_zh.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)

        print("[OK] 回测完成")
        print(f"输出目录: {output_dir}")
        print()
        print(summary_text)
    finally:
        tq.close()


if __name__ == "__main__":
    main()
