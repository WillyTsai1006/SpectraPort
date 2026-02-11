import numpy as np
import pandas as pd
from src.denoise import RMTDenoising, LedoitWolfDenoising

class RollingBacktest:
    """
    滾動視窗回測引擎
    """
    def __init__(self, returns, window=252, rebalance_freq=21):
        self.returns = returns # 這裡必須是 Raw Returns (pct_change)
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.weights_history = {}
        
    def run_strategy(self, name, optimizer, denoise_method=None):
        """
        執行單一策略的回測
        """
        T = len(self.returns)
        portfolio_returns = []
        dates = []
        print(f"Backtesting strategy: {name}...")
        # 滾動迴圈
        for t in range(self.window, T, self.rebalance_freq):
            # 1. 取得訓練窗口 (Train Window)
            train_data = self.returns.iloc[t-self.window : t]
            # 2. 取得測試窗口 (Test Window) - 未來的一個月
            test_data = self.returns.iloc[t : min(t+self.rebalance_freq, T)]
            if test_data.empty: break
            # 3. 計算協方差矩陣 (Covariance Matrix)
            if denoise_method == 'RMT':
                # RMT 需要標準化數據來算 Correlation
                # train_std = (train_data - train_data.mean()) / train_data.std() # RMT 類別內會處理 corr
                rmt = RMTDenoising().fit(train_data) # fit 裡面會算 corr
                corr_clean = rmt.transform()
                # 將去噪後的 Corr 轉回 Cov: Cov = D * Corr * D
                stds = train_data.std().values
                cov_estim = np.outer(stds, stds) * corr_clean
                # 轉為 DataFrame 以便 optimizer 使用 index
                cov_estim = pd.DataFrame(cov_estim, index=train_data.columns, columns=train_data.columns) 
            elif denoise_method == 'LW':
                lw = LedoitWolfDenoising().fit(train_data)
                # LW 的 transform 我們之前寫的是回傳 corr，這裡我們直接拿 cov
                cov_estim = lw.lw.covariance_
                cov_estim = pd.DataFrame(cov_estim, index=train_data.columns, columns=train_data.columns)
            else:
                # 原始 Covariance
                cov_estim = train_data.cov()
            # 4. 計算權重
            try:
                if name == 'HRP':
                    # HRP 直接吃原始 returns
                    w = optimizer.get_weights(train_data)
                else:
                    # GMVP 吃 Covariance
                    w = optimizer.get_gmvp_weights(cov_estim)
            except Exception as e:
                print(f"Optimization failed at {t}: {e}")
                w = pd.Series(1.0/train_data.shape[1], index=train_data.columns)
            # 記錄權重 (方便之後畫圖分析)
            if name not in self.weights_history:
                self.weights_history[name] = []
            self.weights_history[name].append(w)
            # 5. 計算樣本外績效 (Out-of-Sample Return)
            # 假設這一個月內權重不變 (Buy and Hold)
            # Daily Portfolio Return = w * r
            p_ret = test_data.dot(w)
            portfolio_returns.extend(p_ret.values)
            dates.extend(test_data.index)
        return pd.Series(portfolio_returns, index=dates, name=name)