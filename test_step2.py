import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.optimization import MeanVarianceOptimizer, HRPOptimizer
from src.backtest import RollingBacktest

# 1. 載入數據
print("Loading data...")
tickers = [
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'CRM',
    'ADBE', 'CSCO', 'ACN', 'AMD', 'ORCL', 'INTC', 'QCOM', 'TXN', 'IBM', 'AMAT',
    'MU', 'ADI', 'LRCX', 'NOW', 'ADP', 'FISV', 'KLAC', 'SNPS', 'CDNS', 'ROP',
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SCHW', 'C',
    'BLK', 'SPGI', 'AXP', 'PGR', 'CB', 'MMC', 'USB', 'PNC', 'TFC', 'BK',
    'AON', 'CME', 'ICE', 'MCO', 'COF', 'MET', 'AIG', 'TRV', 'ALL', 'PRU',
    'UNH', 'JNJ', 'LLY', 'MRK', 'ABBV', 'PFE', 'TMO', 'DHR', 'ABT', 'BMY',
    'AMGN', 'CVS', 'ELV', 'ISRG', 'MDT', 'GILD', 'SYK', 'CI', 'REGN', 'VRTX',
    'ZTS', 'BDX', 'BSX', 'HUM', 'EW', 'HCA', 'MCK', 'CNC', 'IQV', 'BAX',
    'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'TGT', 'BKNG', 'F', 'GM',
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'EL', 'CL', 'MDLZ',
    'KHC', 'GIS', 'SYY', 'STZ', 'K', 'HSY', 'CLX', 'KMB', 'DG', 'DLTR',
    'CAT', 'DE', 'HON', 'UNP', 'UPS', 'GE', 'BA', 'LMT', 'RTX', 'MMM',
    'ETN', 'ITW', 'WM', 'NSC', 'CSX', 'EMR', 'GD', 'FDX', 'NOC', 'PH',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'PEG',
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB',
    'LIN', 'SHW', 'APD', 'FCX', 'ECL', 'NEM', 'DOW', 'CTVA', 'DD', 'PPG'
]
loader = DataLoader(tickers, start_date='2021-01-01', end_date='2023-12-30') # 抓長一點 3年
prices = loader.fetch_data()
returns = loader.get_returns(prices)
# 2. 初始化回測引擎 (Window=252天, Rebalance=21天)
bt = RollingBacktest(returns, window=252, rebalance_freq=21)
mv_opt = MeanVarianceOptimizer(long_only=False) # 允許放空
hrp_opt = HRPOptimizer()
# 3. 執行策略
results = pd.DataFrame()
# Benchmark: Equal Weight
ew_ret = returns.mean(axis=1) # 簡單平均
ew_ret.name = 'Equal Weight'
# 為了對齊時間，我們只取回測開始後的時間段
start_date = bt.run_strategy('Dummy', mv_opt).index[0] 
results['Equal Weight'] = ew_ret.loc[start_date:]
# Strategy 1: Raw GMVP
results['Raw GMVP'] = bt.run_strategy('Raw GMVP', mv_opt, denoise_method=None)
# Strategy 2: RMT GMVP
results['RMT GMVP'] = bt.run_strategy('RMT GMVP', mv_opt, denoise_method='RMT')
# Strategy 3: LW GMVP
results['LW GMVP'] = bt.run_strategy('LW GMVP', mv_opt, denoise_method='LW')
# Strategy 4: HRP
results['HRP'] = bt.run_strategy('HRP', hrp_opt)
# 4. 顯示績效指標
print("\n=== Backtest Results (Annualized) ===")
stats = pd.DataFrame()
for col in results.columns:
    ann_ret = results[col].mean() * 252
    ann_vol = results[col].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    stats.loc[col, 'Return'] = f"{ann_ret*100:.2f}%"
    stats.loc[col, 'Volatility'] = f"{ann_vol*100:.2f}%"
    stats.loc[col, 'Sharpe'] = f"{sharpe:.2f}"
print(stats)
# 5. 畫累積報酬圖
(1 + results).cumprod().plot(figsize=(10, 6))
plt.title("Out-of-Sample Cumulative Returns")
plt.show()
# 6. 畫權重比較 (取最後一個調倉日的權重)
last_weights_raw = bt.weights_history['Raw GMVP'][-1]
last_weights_rmt = bt.weights_history['RMT GMVP'][-1]
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
last_weights_raw.plot(kind='bar')
plt.title("Raw GMVP Weights (Unstable)")
plt.xticks([]) # 隱藏 x 軸標籤以免太擠
plt.subplot(1, 2, 2)
last_weights_rmt.plot(kind='bar')
plt.title("RMT GMVP Weights (Stabilized)")
plt.xticks([])
plt.show()