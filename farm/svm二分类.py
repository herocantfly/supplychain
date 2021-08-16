# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:36:23 2021

@author: LindaHK
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#import mglearn.datasets
import matplotlib.pyplot as plt
#forge数据集是一个二维二分类数据集
data=pd.read_csv(r'C:\Users\LindaHK\Desktop\farm\data1.csv')
#X,y=data.iloc[:,:-1],data.iloc[:,-1]
#print(X,y)
'''
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=33)
svm=SVC(kernel='rbf',C=10,gamma=0.1,probability=True).fit(X_train,y_train)
 
print(svm.predict(X_test))
#输出分类概率
print(svm.predict_proba(X_test))
print(svm.score(X_test,y_test))
'''