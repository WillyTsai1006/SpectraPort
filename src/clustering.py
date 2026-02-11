import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

class HierarchicalClustering:
    """
    實作階層風險平價 (HRP) 所需的分群與排序演算法
    """
    @staticmethod
    def get_linkage(corr):
        """產生連結矩陣 (Linkage Matrix)"""
        # 距離定義：d = sqrt(0.5 * (1 - rho))
        dist = np.sqrt(0.5 * (1 - corr))
        link = sch.linkage(squareform(dist), 'single')
        return link

    @staticmethod
    def get_quasi_diag(link):
        """
        Quasi-Diagonalization: 重排矩陣順序，將相似資產放在一起
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])         
        return sort_ix.tolist()

    @staticmethod
    def get_rec_bisection(cov, sort_ix):
        """
        遞迴二分法 (Recursive Bisection): 自上而下分配權重
        """
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]   # 左子樹
                c_items1 = c_items[i + 1] # 右子樹
                c_var0 = HierarchicalClustering._get_cluster_var(cov, c_items0)
                c_var1 = HierarchicalClustering._get_cluster_var(cov, c_items1)
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha     
        return w

    @staticmethod
    def _get_cluster_var(cov, c_items):
        """計算群組內的變異數 (Inverse Variance Allocation)"""
        cov_slice = cov.loc[c_items, c_items]
        # 計算群組內各資產的IVP權重
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        # V_cluster = w^T * Cov * w
        w = ivp.reshape(-1, 1)
        c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
        return c_var