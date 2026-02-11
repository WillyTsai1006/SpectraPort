import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf

# 設定繪圖風格
sns.set_style('whitegrid')

class RMTDenoising:
    """
    基於隨機矩陣理論 (RMT) 的去噪器。
    參考 Marchenko-Pastur 定律來過濾雜訊特徵值。
    """
    def __init__(self, alpha=2.0):
        self.alpha = alpha
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.lambda_max = None
        self.lambda_min = None
        self.q = None
        self.sigma2 = 1
    
    def fit(self, X):
        """
        計算特徵值並擬合 MP 分佈。
        X: 標準化後的收益率矩陣 (T x N) 或 DataFrame
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        T, N = X.shape
        self.q = T / N
        # 1. 計算經驗相關係數矩陣
        self.corr_matrix = np.corrcoef(X, rowvar=False)
        # 2. 特徵值分解
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eigh(self.corr_matrix)
        # 3. 計算 Marchenko-Pastur 界限（完整公式）
        self.lambda_max = self.sigma2 * (1.0 + np.sqrt(1.0/self.q))**2
        self.lambda_min = self.sigma2 * (1.0 - np.sqrt(1.0/self.q))**2
        return self

    def transform(self, X=None):
        """執行去噪並重建矩陣"""
        if self.eigenvalues_ is None:
            raise ValueError("Run .fit() first!")
        # 將超過 lambda_max 的特徵值保留，其餘替換為雜訊平均
        n_signals = np.sum(self.eigenvalues_ > self.lambda_max)
        eigenvalues_clean = self.eigenvalues_.copy()
        if n_signals < len(eigenvalues_clean):
            # 計算雜訊部分的平均特徵值
            noise_vars = eigenvalues_clean[:-n_signals]
            noise_mean = np.mean(noise_vars)
            eigenvalues_clean[:-n_signals] = noise_mean
        # 重建相關係數矩陣
        corr_clean_raw = self.eigenvectors_ @ np.diag(eigenvalues_clean) @ self.eigenvectors_.T
        # 對角線正規化 (設回 1)
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(corr_clean_raw)))
        corr_clean = D_inv_sqrt @ corr_clean_raw @ D_inv_sqrt
        return corr_clean

    def plot_spectrum(self):
        """繪製特徵值分佈 vs MP 理論曲線"""
        if self.eigenvalues_ is None:
            raise ValueError("Run .fit() first!")
        # 產生理論 PDF
        x = np.linspace(self.lambda_min, self.lambda_max, 1000)
        pdf = (self.q / (2 * np.pi * self.sigma2 * x)) * np.sqrt((self.lambda_max - x) * (x - self.lambda_min))
        pdf = np.nan_to_num(pdf)
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf, color='orange', linewidth=3, label='Theoretical MP Law (Noise)')
        # 為了視覺美觀，過濾掉超級大的 Market Mode 特徵值
        evals_display = self.eigenvalues_[self.eigenvalues_ <= self.lambda_max * self.alpha]
        plt.hist(evals_display, bins=50, density=True, alpha=0.6, color='steelblue', label='Empirical Eigenvalues')
        plt.axvline(self.lambda_max, color='red', linestyle='--', label=f'Noise Upper Bound ({self.lambda_max:.2f})')
        plt.title(f'Eigenvalue Spectrum (Q={self.q:.2f})')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

class LedoitWolfDenoising:
    """
    Ledoit-Wolf 收縮估計 (對照組)
    """
    def __init__(self):
        self.lw = LedoitWolf()
        
    def fit(self, X):
        self.lw.fit(X)
        return self
        
    def transform(self):
        # 取得協方差矩陣
        cov = self.lw.covariance_
        # 轉為相關係數矩陣 (以便與 RMT 比較條件數)
        d = np.diag(cov)
        std_dev = np.sqrt(d)
        corr_lw = cov / np.outer(std_dev, std_dev)
        return corr_lw