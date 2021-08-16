# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:01:37 2021

@author: LindaHK
"""
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


from sklearn.neural_network import MLPRegressor
data=pd.read_csv(r'C:\Users\LindaHK\Desktop\DATA.csv',index_col='CODE')
X=data.iloc[:,0:-1]
Y=data.iloc[:,-1:]
print(X)
print(Y)
zscore = preprocessing.StandardScaler()
# 标准化处理
data_zs = zscore.fit_transform(X)
print(data_zs)
label_zs = zscore.fit_transform(Y)

#dummies = pd.get_dummies(data['Y'],prefix='Y')
#print(dummies)
X_train=data_zs[:-8]
X_test=data_zs[-8:]
Y_train=label_zs[:-8]
Y_test=label_zs[-8:]

model = MLPRegressor(hidden_layer_sizes=(3,), random_state=10,learning_rate_init=0.01)  # BP神经网络回归模型
model.fit(X_train,Y_train)  # 训练模型
pre = model.predict(X_test)  
pre_t=zscore.inverse_transform(pre)
# 模型预测
predict=np.abs(Y_test-pre).mean()  # 模型评价

# 衡量线性回归的MSE 、 RMSE、 MAE、r2
print("mean_absolute_error:", mean_absolute_error(Y_test,pre))
print("mean_squared_error:", mean_squared_error(Y_test,pre))
print("rmse:", sqrt(mean_squared_error(Y_test,pre)))
print("r2 score:", r2_score(Y_test,pre))
pre2_t = zscore.inverse_transform(model.predict(X_train))
print(predict)
print(zscore.inverse_transform(pre))
print(pre2_t)


