import argparse
from datetime import datetime

import pandas as pd

from tqcenter import tq
from integrated_tq_strategy import AdvancedShortTermStrategyV6


def parse_args():
    parser = argparse.ArgumentParser(description="通达信专用选股脚本：只运行一次选股并退出")
    parser.add_argument("--block-name", default="沪深300", help="策略显示用板块名称")
    parser.add_argument("--lookback-days", type=int, default=40, help="选股回看天数下限")
    parser.add_argument("--as-of-date", default="", help="指定选股日期，格式 YYYYMMDD；为空则默认今天")
    parser.add_argument("--top-k", type=int, default=0, help="仅保留前 K 名，0 表示不限制数量")
    parser.add_argument("--target-tdx-block", default="威科夫共振", help="推送到通达信的自定义板块名称")
    parser.add_argument("--show-top", type=int, default=20, help="控制台展示前 N 名")
    parser.add_argument("--output-csv", default="", help="将完整选股结果导出到 CSV 文件")
    parser.add_argument("--push-block", action="store_true", help="将结果推送到通达信自定义板块")

    market_risk_group = parser.add_mutually_exclusive_group()
    market_risk_group.add_argument("--market-risk", dest="market_risk_enabled", action="store_true", help="开启大盘风控拦截")
    market_risk_group.add_argument("--no-market-risk", dest="market_risk_enabled", action="store_false", help="关闭大盘风控拦截")
    parser.set_defaults(market_risk_enabled=False)
    return parser.parse_args()


def build_strategy(args):
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    return AdvancedShortTermStrategyV6(
        block_name=args.block_name,
        lookback_days=args.lookback_days,
        top_k=top_k,
        target_tdx_block=args.target_tdx_block,
        market_risk_enabled=args.market_risk_enabled,
        enable_logs=True,
    )


def resolve_as_of_date(raw_date):
    if raw_date:
        target_dt = datetime.strptime(raw_date, "%Y%m%d")
    else:
        target_dt = datetime.now()

    query_start = (target_dt - pd.Timedelta(days=31)).strftime("%Y%m%d")
    query_end = target_dt.strftime("%Y%m%d")
    trading_days = tq.get_trading_dates(market="SH", start_time=query_start, end_time=query_end, count=-1) or []
    if not trading_days:
        raise ValueError("无法获取交易日历。")

    target_str = target_dt.strftime("%Y%m%d")
    valid_days = [day for day in trading_days if day <= target_str]
    if not valid_days:
        raise ValueError(f"在 {target_str} 之前未找到可用交易日。")
    return valid_days[-1]


def fetch_history_for_date(strategy, as_of_date):
    fetch_count = max(strategy.lookback_days, 520)
    end_dt = datetime.strptime(as_of_date, "%Y%m%d")
    start_date = (end_dt - pd.Timedelta(days=fetch_count * 2)).strftime("%Y%m%d")
    stocks = strategy._get_universe()
    all_to_fetch = list(set(stocks + ["999999.SH"]))
    return tq.get_market_data(
        field_list=["Close", "Volume", "Amount", "Open", "High", "Low"],
        stock_list=all_to_fetch,
        start_time=start_date,
        end_time=as_of_date,
        count=fetch_count,
        dividend_type="front",
        period="1d",
    )


def run_selector(strategy, as_of_date, push_block=False):
    df_dict = fetch_history_for_date(strategy, as_of_date)
    ranked_df = strategy.select_from_history(df_dict, execute_signals=False)

    strategy.my_positions = ranked_df["Close"].to_dict() if not ranked_df.empty else {}

    if push_block and not ranked_df.empty:
        strategy._execute_signals(ranked_df)

    return ranked_df


def main():
    args = parse_args()
    print("=" * 60)
    print(">>> 启动专用选股脚本：运行一次并退出 <<<")
    print("=" * 60)

    tq.initialize(__file__)
    try:
        strategy = build_strategy(args)
        as_of_date = resolve_as_of_date(args.as_of_date)
        ranked_df = run_selector(strategy, as_of_date=as_of_date, push_block=args.push_block)

        print(f"\n[V] 选股基准日期: {as_of_date}")

        if ranked_df.empty:
            print("\n[-] 本次未选出符合条件的标的。")
            return

        print(f"\n[+] 本次共选出 {len(ranked_df)} 只标的。")
        show_n = max(int(args.show_top), 0)
        if show_n > 0:
            print(f"\n前 {min(show_n, len(ranked_df))} 名：")
            print(ranked_df[["Total_Score", "Close", "Signal_Tag"]].head(show_n).to_string())

        if args.output_csv:
            ranked_df.to_csv(args.output_csv, encoding="utf-8-sig")
            print(f"\n[V] 结果已导出到: {args.output_csv}")

        if args.push_block:
            print(f"\n[V] 结果已推送到通达信自定义板块: {args.target_tdx_block}")
        else:
            print("\n[V] 本次仅输出结果，未推送到通达信板块。")
    except KeyboardInterrupt:
        print("\n[!] 用户手动中止。")
    except Exception as exc:
        print(f"\n[X] 选股脚本运行失败: {exc}")
    finally:
        tq.close()
        print(f"[V] 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
