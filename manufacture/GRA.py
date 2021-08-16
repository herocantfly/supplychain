# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:06:52 2021

@author: LindaHK
"""
#解决中文显示问题

import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 从硬盘读取数据进入内存
data = pd.read_csv(r'C:\Users\LindaHK\Desktop\DATA.csv')
data1=data.drop(['Y1'],axis=1)*100
data1['X11']=data1['X11']/100
y=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','Y1']
data1['X1']=data1['X1'].apply(int)
data1['X2']=data1['X2'].apply(int)
data1['X3']=data1['X3'].apply(int)
data1['X4']=data1['X4'].apply(int)
data1['X5']=data1['X5'].apply(int)
data1['X6']=data1['X6'].apply(int)
data1['X7']=data1['X7'].apply(int)
data1['X8']=data1['X8'].apply(int)
data1['X9']=data1['X9'].apply(int)
data1['X10']=data1['X10'].apply(int)
data1['X11']=data1['X11'].apply(int)
'''data1['X12']=data1['X12'].apply(int)
data1['X13']=data1['X13'].apply(int)
data1['X14']=data1['X14'].apply(int)
data1['X15']=data1['X15'].apply(int)
data1['X16']=data1['X16'].apply(int)
data1['X17']=data1['X17'].apply(int)
data1['X18']=data1['X18'].apply(int)
data1['X19']=data1['X19'].apply(int)'''
#data1['Y']=data['Y'].apply(int)
print(data1)

def GRA_ONE(gray, m=0):
    # 读取为df格式
    gray = gray - gray.min()/ gray.max() - gray.min()
    # 标准化
    std= gray.iloc[:, m]  # 为标准要素
    ce = gray.iloc[:, 0:]  # 为比较要素
    n, m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c, d = np.amax(a), np.amin(a)

    # 计算值
    result = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值,并返回
    return pd.DataFrame([np.mean(result[i, :]) for i in range(m)])


def GRA(DataFrame):
    list_columns = [
        str(s) for s in range(len(DataFrame.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    for i in range(len(DataFrame.columns)):
        df_local.iloc[:, i] = GRA_ONE(DataFrame, m=i)[0]
    return df_local

data_gra = GRA_ONE(data1)
path=r'C:\Users\LindaHK\Desktop\manufacture'
data_gra.to_csv(path+"GRA.csv") 
print(data_gra)

# 灰色关联结果矩阵可视化
import seaborn as sns


def ShowGRAHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    f, ax = plt.subplots(figsize=(14, 10.5))
    ax.set_title('指标关联度矩阵')
    sns.heatmap(DataFrame.astype(float),
                cmap=colormap,
                ax=ax,
                annot=True,
                yticklabels=14,
                xticklabels=10)
    plt.show()


ShowGRAHeatMap(data_gra)
