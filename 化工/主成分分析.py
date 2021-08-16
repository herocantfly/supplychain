# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:29:30 2021

@author: LindaHK
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np


df = pd.read_csv(r'C:\Users\LindaHK\Desktop\DATA.csv') # 读取数据
data = scale(df.values) # 标准化，标准化之后就自动根据协方差矩阵进行主成分分析了
# data2 = np.corrcoef(np.transpose(data)) # 没有必要单独计算协方差阵或相关系数阵
pca = PCA() # 可以调整主成分个数，n_components = 1
pca.fit(data)
print(pca.explained_variance_) # 输出特征根
print(pca.explained_variance_ratio_) # 输出解释方差比
print(pca.components_) # 输出主成分
