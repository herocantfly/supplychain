# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:17:43 2021

@author: LindaHK
"""
#数据预处理（标准化与分类数据独热编码）
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\LindaHK\Desktop\DATA.csv',index_col='CODE')
X=data.iloc[:,0:-1]
Y=data.iloc[:,-1:]
print(X)
print(Y)
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs = zscore.fit_transform(X)
print(data_zs)

dummies = pd.get_dummies(data['Y'],prefix='Y')
print(dummies)
X_train=data_zs[:-8]
X_test=data_zs[-8:]
Y_train=dummies[:-8]
Y_test=dummies[-8:]