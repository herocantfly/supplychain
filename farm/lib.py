# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:31:48 2021

@author: LindaHK
"""

#测试代码：（注意libsvm-3.23/python路径）
import sys
path=r'C:\Users\LindaHK\Desktop\libsvm-3.25\python'
sys.path.append(path)
import os
os.chdir(path)
from svmutil import *
train_label,train_pixel = svm_read_problem(r'C:\Users\LindaHK\Desktop\farm\farmdata.txt')
practical_label,predict_pixel = svm_read_problem(r'C:\Users\LindaHK\Desktop\farm\farmdata.txt')
model = svm_train(train_label,train_pixel,'-s 0 -t 2 -c 1 -g 0.8')
print("result:")
p_label, p_acc, p_val = svm_predict(practical_label, predict_pixel, model);
print(p_acc)
print(practical_label)
print(p_label)
