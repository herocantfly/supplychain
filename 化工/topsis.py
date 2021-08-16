# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:10:46 2021

@author: LindaHK
"""

import pandas as pd
import numpy as np


def topsis(data, weight=None):
# 归一化
    data = data / np.sqrt((data ** 2).sum())

# 最优最劣方案
    Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

# 距离
    weight = entropyWeight(data) if weight is None else np.array(weight)
    Result = data.copy()
    Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
    Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

# 综合得分指数
    Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解']+Result['正理想解'])                                           #     + Result['正理想解'])
    Result['排序'] = Result.rank(ascending=False)['综合得分指数']#综合得分的地方有错误

    return Result, Z, weight
data= pd.read_csv(r'C:\Users\LindaHK\Desktop\DATA.csv', encoding='gb2312')
weight=[0.025715,0.027849,0.016109,0.212281,0.056274,0.032895,0.036161,0.037516,0.007308,0.045387,0.025521,0.476985]
result,z,weights=topsis(data,weight)
result.to_csv(r'C:\Users\LindaHK\Desktop\result.csv')
z.to_csv(r'C:\Users\LindaHK\Desktop\z.csv')
#weights.to_csv(r'C:\Users\LindaHK\Desktop\weights.csv')
print(z,weights)