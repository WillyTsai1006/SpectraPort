import numpy as np
import pandas as pd
from src.clustering import HierarchicalClustering

class MeanVarianceOptimizer:
    """
    傳統均值-變異數優化 (Markowitz)
    """
    def __init__(self, long_only=False):
        self.long_only = long_only # 預設 False 以展示原始矩陣的不穩定性

    def get_gmvp_weights(self, cov_matrix):
        """
        計算全域最小變異數組合 (GMVP)
        w = (Sigma^-1 * 1) / (1^T * Sigma^-1 * 1)
        """
        # 如果是 DataFrame，轉 numpy，但要記錄 index
        if isinstance(cov_matrix, pd.DataFrame):
            assets = cov_matrix.index
            cov = cov_matrix.values
        else:
            assets = range(cov_matrix.shape[0])
            cov = cov_matrix
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(len(cov))
            w = np.dot(inv_cov, ones) / np.dot(np.dot(ones.T, inv_cov), ones)
            return pd.Series(w, index=assets)
        except np.linalg.LinAlgError:
            # 如果矩陣真的不可逆 (極少發生)，回傳等權重
            return pd.Series(np.ones(len(cov))/len(cov), index=assets)

class HRPOptimizer:
    """
    階層風險平價優化 (HRP)
    """
    def get_weights(self, returns):
        # 1. 準備數據
        corr = returns.corr()
        cov = returns.cov()
        # 2. 分群與排序
        hc = HierarchicalClustering()
        link = hc.get_linkage(corr.values)
        sort_ix_list = hc.get_quasi_diag(link)
        # 轉換回 index 名稱
        sort_ix = corr.index[sort_ix_list].tolist()
        # 3. 遞迴分配權重
        # 注意：重排 covariance matrix
        cov_sorted = cov.loc[sort_ix, sort_ix]
        weights = hc.get_rec_bisection(cov_sorted, sort_ix)
        # 轉回原始順序
        return weights.loc[returns.columns]