# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:20:57 2021

@author: LindaHK
"""

"""
Created on Fri Dec 13 00:56:39 2019

@author: LindaHK
"""
from svmutil import *
import numpy as np
from numpy import *
import pandas as pd
import random
from time import *
import os
import sys
os.chdir(r'C:\Users\LindaHK\Desktop\libsvm-3.25\libsvm-3.25\python')
from svmutil import*
data=pd.read_csv(r'C:\Users\LindaHK\Desktop\farm\data1.csv')
x,y=data.iloc[:,:-1],data.iloc[:,-1]
#print(X,y)
#y,x=svm_read_problem(r'C:\Users\LindaHK\Desktop\desktop\data5.txt')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)
prob = svm_problem(y_train, x_train)
param = svm_parameter('-t 3 -c 4 -b 1')
model = svm_train(prob, param)
model
'''-s～设置svm类型：0 – C-SVC/1 – v-SVC/2 – one-class-SVM/3 –ε-SVR/4 – n - SVR

-t～设置核函数类型，默认值为2：0 --线性核：u'*v/1 --多项式核：(g*u'*v+coef0)degree/2 -- RBF核：exp(-γ*||u-v||2)/3 -- sigmoid核：tanh(γ*u'*v+coef0)

-d～degree:设置多项式核中degree的值，默认为3

-g～γ:设置核函数中γ的值，默认为1/k，k为特征（或者说是属性）数；

-r～coef 0:设置核函数中的coef 0，默认值为0；

-c～cost：设置C-SVC、ε-SVR、n - SVR中从惩罚系数C，默认值为1；

-n～v：设置v-SVC、one-class-SVM与n - SVR中参数n，默认值0.5；

-p～ε：设置v-SVR的损失函数中的e，默认值为0.1；

-m～cachesize：设置cache内存大小，以MB为单位，默认值为40；

-e～ε：设置终止准则中的可容忍偏差，默认值为0.001；

-h～shrinking：是否使用启发式，可选值为0或1，默认值为1；

-b～概率估计：是否计算SVC或SVR的概率估计，可选值0或1，默认0；

-wi～weight：对各类样本的惩罚系数C加权，默认值为1；

-v～n：n折交叉验证模式；
'''
p_label, p_acc, p_val = svm_predict(y_test, x_test, model)

print(p_label)
print(p_acc)
print(p_val)
