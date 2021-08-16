# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:18:32 2021

@author: LindaHK
"""

import pandas as pd
rawdata = pd.read_csv(r'C:\Users\LindaHK\Desktop\农业\DATA.csv') 
X = rawdata.iloc[:,:-2]
Y = rawdata.iloc[:,-2]  # {”A":0,"B":1,"C":2}
Y = pd.Categorical(Y).codes  # ABC变成123
print(X,Y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import sklearn.svm as svm
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

model = svm.SVC(kernel="linear", decision_function_shape="ovo")
model = model.fit(x_train, x_train)

acu_train = model.score(x_train, y_train)
acu_test = model.score(x_test, y_test)
y_pred = model.predict(x_test)
recall = recall_score(y_test, y_pred, average="macro")
