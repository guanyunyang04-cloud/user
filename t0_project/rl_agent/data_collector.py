import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqcenter import tq

def collect_training_data(stock_list, days_back=30, period='1m', output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("==================================================")
    print(f">>> Starting A-Share {period} Historical Data Collection <<<")
    print(f"Target Stocks: {stock_list}")
    print(f"Lookback Days: {days_back}")
    print("==================================================")

    tq.initialize(__file__)

    try:
        # We verified that the TDX client has 1m data up to 30 days (7200 bars)
        if period == '1m':
            count = days_back * 240
        elif period == '5m':
            count = days_back * 48
        else:
            count = days_back # '1d'

        print(f"[*] Requesting data from TDX server (approx {count} bars per stock)...")
        
        df_dict = tq.get_market_data(
            field_list=['Open', 'High', 'Low', 'Close', 'Volume', 'Amount'],
            stock_list=stock_list,
            start_time='', 
            end_time='',
            count=count,
            dividend_type='front',
            period=period
        )

        if not df_dict or 'Close' not in df_dict or df_dict['Close'].empty:
            print("[-] Data fetch failed, server returned empty data.")
            return

        saved_files = []
        for stock in stock_list:
            if stock not in df_dict['Close'].columns:
                print(f"    [!] Failed to get data for {stock}, skipping.")
                continue

            stock_df = pd.DataFrame({
                'Open': df_dict['Open'][stock],
                'High': df_dict['High'][stock],
                'Low': df_dict['Low'][stock],
                'Close': df_dict['Close'][stock],
                'Volume': df_dict['Volume'][stock],
                'Amount': df_dict['Amount'][stock]
            })
            
            stock_df.dropna(how='all', inplace=True)
            
            if stock_df.empty:
                print(f"    [!] {stock} data is entirely empty, skipping.")
                continue

            filename = f"{stock}_{period}_history.csv"
            filepath = os.path.join(output_dir, filename)
            
            stock_df.index.name = 'Datetime'
            stock_df.to_csv(filepath)
            saved_files.append(filepath)
            print(f"    [+] Successfully exported {stock} -> {len(stock_df)} bars, saved to {filename}")

        print("\n[V] Data collection task completed!")
        print(f"Saved {len(saved_files)} data files for RL training environment.")

    except Exception as e:
        print(f"[-] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tq.close()
        print("TDX connection safely closed.")

if __name__ == "__main__":
    target_stocks = [
        '600536.SH' # 中国软件 - Highly volatile tech stock, good for T+0
    ]
    
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    
    # Switch to 1m data for 30 days as verified available
    collect_training_data(
        stock_list=target_stocks, 
        days_back=30, 
        period='1m', 
        output_dir=data_directory
    )