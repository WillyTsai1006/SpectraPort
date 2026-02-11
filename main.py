import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.denoise import RMTDenoising, LedoitWolfDenoising
from src.optimization import MeanVarianceOptimizer, HRPOptimizer
from src.backtest import RollingBacktest

# 確保圖片存檔目錄存在
if not os.path.exists('images'):
    os.makedirs('images')

def main():
    print("=== EigenRisk: Portfolio Optimization Framework ===")
    # 1. 設定與載入數據
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
    print("1. Loading Data...")
    loader = DataLoader(tickers, start_date='2021-01-01', end_date='2023-12-30')
    prices = loader.fetch_data()
    returns = loader.get_returns(prices)
    print(f"   Data Shape: {returns.shape}")
    # 2. 展示 RMT 頻譜去噪效果 (Physics View)
    print("2. Generating RMT Spectrum Analysis...")
    rmt = RMTDenoising().fit(returns.iloc[:252]) # 用第一年數據做展示
    rmt.plot_spectrum() 
    # 3. 執行滾動回測 (Strategy View)
    print("3. Running Rolling Window Backtest (Window=252, Rebalance=21)...")
    bt = RollingBacktest(returns, window=252, rebalance_freq=21)
    mv_opt = MeanVarianceOptimizer(long_only=False) # 展示無限制的原始特性
    hrp_opt = HRPOptimizer()
    results = pd.DataFrame()
    # Benchmark
    ew = returns.mean(axis=1)
    ew.name = 'Equal Weight'
    start_date = bt.run_strategy('Dummy', mv_opt).index[0]
    results['Equal Weight'] = ew.loc[start_date:]
    # Strategies
    results['Raw GMVP'] = bt.run_strategy('Raw GMVP', mv_opt, denoise_method=None)
    results['RMT GMVP'] = bt.run_strategy('RMT GMVP', mv_opt, denoise_method='RMT')
    results['LW GMVP'] = bt.run_strategy('LW GMVP', mv_opt, denoise_method='LW')
    results['HRP'] = bt.run_strategy('HRP', hrp_opt)
    # 4. 計算並儲存績效表
    print("\n=== Performance Metrics ===")
    metrics = []
    for col in results.columns:
        ann_ret = results[col].mean() * 252
        ann_vol = results[col].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        max_dd = (results[col] + 1).cumprod().div((results[col] + 1).cumprod().cummax()).sub(1).min()
        metrics.append({
            'Strategy': col,
            'Return': ann_ret,
            'Volatility': ann_vol,
            'Sharpe': sharpe,
            'Max Drawdown': max_dd
        })
    metrics_df = pd.DataFrame(metrics).set_index('Strategy')
    # 格式化輸出
    print(metrics_df.style.format("{:.2%}").to_string())
    metrics_df.to_csv('images/metrics.csv')
    # 5. 繪製並儲存累積報酬圖
    plt.figure(figsize=(12, 6))
    (1 + results).cumprod().plot(linewidth=2)
    plt.title("Out-of-Sample Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.savefig('images/cumulative_returns.png', dpi=300)
    print("\n[Saved] images/cumulative_returns.png")
    # 6. 繪製權重比較圖
    # 取最後一次調倉的權重
    w_raw = bt.weights_history['Raw GMVP'][-1]
    w_rmt = bt.weights_history['RMT GMVP'][-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    w_raw.plot(kind='bar', ax=axes[0], color='salmon', width=1.0)
    axes[0].set_title(f"Raw GMVP Weights\n(Cond #: {np.linalg.cond(returns.corr()):.0f})")
    axes[0].set_xlabel("Assets")
    axes[0].set_xticks([]) # 隱藏 x 軸標籤
    w_rmt.plot(kind='bar', ax=axes[1], color='steelblue', width=1.0)
    axes[1].set_title(f"RMT GMVP Weights\n(Denoised)")
    axes[1].set_xlabel("Assets")
    axes[1].set_xticks([])
    plt.tight_layout()
    plt.savefig('images/weights_comparison.png', dpi=300)
    print("[Saved] images/weights_comparison.png")
    print("\nDone! Check the 'images' folder.")

if __name__ == "__main__":
    main()