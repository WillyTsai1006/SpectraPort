import yfinance as yf
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, tickers, start_date, end_date, data_dir='data'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def fetch_data(self):
        """下載或讀取快取數據"""
        file_path = os.path.join(self.data_dir, 'stock_prices.csv')
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print("Downloading data from Yahoo Finance...")
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
            # 簡單清洗：刪除缺失值過多的股票
            df = df.dropna(axis=1, thresh=int(len(df)*0.9)) 
            df = df = df.ffill().bfill()
            df.to_csv(file_path)
        return df

    def get_returns(self, df):
        """計算百分比報酬率 (Raw Returns)"""
        # 這裡不標準化，保留原始波動，供回測使用
        return df.pct_change().dropna()