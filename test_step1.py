from src.data_loader import DataLoader
from src.denoise import RMTDenoising, LedoitWolfDenoising
import numpy as np

# 1. 設定你的 180 檔股票代碼 (這裡用你的列表，為求簡潔我放幾個代表)
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
# 2. 下載數據
print("Step 1: Loading Data...")
loader = DataLoader(tickers, start_date='2022-01-01', end_date='2023-12-30')
prices = loader.fetch_data()
returns = loader.get_returns(prices)
print(f"Data shape: {returns.shape}")

# 3. 測試 RMT
print("\nStep 2: Testing RMT Denoising...")
rmt = RMTDenoising()
rmt.fit(returns)
corr_rmt = rmt.transform()

# 顯示 RMT 頻譜圖 (確認 Physics View)
print("Plotting Spectrum...")
rmt.plot_spectrum()

# 4. 測試 Ledoit-Wolf
print("\nStep 3: Testing Ledoit-Wolf...")
lw = LedoitWolfDenoising()
lw.fit(returns)
corr_lw = lw.transform()

# 5. 比較條件數 (數學驗證)
raw_cond = np.linalg.cond(returns.corr())
rmt_cond = np.linalg.cond(corr_rmt)
lw_cond = np.linalg.cond(corr_lw)

print(f"\nCondition Number Comparison:")
print(f"Raw Matrix : {raw_cond:,.2f}")
print(f"RMT Matrix : {rmt_cond:,.2f}")
print(f"LW Matrix  : {lw_cond:,.2f}")

if rmt_cond < raw_cond / 10:
    print("\n✅ 第一階段成功！RMT 有效修復了矩陣條件數。")
else:
    print("\n⚠️ 警告：條件數改善不明顯，請檢查數據。")