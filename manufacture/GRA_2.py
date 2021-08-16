# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:14:11 2021

@author: LindaHK
"""

import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
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

# 无量纲化
def dimensionlessProcessing(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        MEAN = d.mean()
        newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()
    return newDataFrame

def GRA_ONE(gray, m=0):
    # 读取为df格式
    gray = dimensionlessProcessing(gray)
    # 标准化
    std = gray.iloc[:, m]  # 为标准要素
    gray.drop(str(m),axis=1,inplace=True)
    ce = gray.iloc[:, 0:]  # 为比较要素
    shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c, d = amax(a), amin(a)

    # 计算值
    result = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值,并返回
    result_list = [mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m,1)
    return pd.DataFrame(result_list)


def GRA(DataFrame):
    df = DataFrame.copy()
    list_columns = [
        str(s) for s in range(len(df.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    df.columns=list_columns
    for i in range(len(df.columns)):
        df_local.iloc[:, i] = GRA_ONE(df, m=i)[0]
    return df_local
# 灰色关联结果矩阵可视化
# 灰色关联结果矩阵可视化
import seaborn as sns

def ShowGRAHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(14, 14))
    ax.set_title('灰色关联度矩阵')
    
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True
    
    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    mask=mask,
                   )
    plt.show()
    
data_gra = GRA(data1)
data_gra.to_csv(r'C:\Users\LindaHK\Desktop\manufacture\GRA.csv') 
ShowGRAHeatMap(data_gra)
