#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 05:54:32 2020

@author: liuyan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
import joblib

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
def generate_X_y(pairs_file): #add feature
    X, y, pairs = [], [], []
    with open(pairs_file, 'r') as fin:
        i = 0
        while True:
            if not (i+1) % 100:
                print('.', end='', file=sys.stderr)
            if not (i+1) % 10000:
                print(i+1, file=sys.stderr)
            i += 1
            line = fin.readline()
            if not line:
                break
            if line[0] != '#':
                sys.exit('ERROR: wrong line!!!')
            pairs.append(line[1:-1])
            cols = fin.readline().split()
            X.append([float(_.split(':')[-1]) for _ in cols[1:]])
            y.append(1 if cols[0] == '+1' else 0)
        print(file=sys.stderr)
    return np.array(X), np.array(y), pairs
    
def generate_X_y2(pairs_file): #add feature
    X, y, pairs = [], [], []
    with open(pairs_file, 'r') as fin:
        i = 0
        while True:
            if not (i+1) % 100:
                print('.', end='', file=sys.stderr)
            if not (i+1) % 10000:
                print(i+1, file=sys.stderr)
            i += 1
            line = fin.readline()
            if not line:
                break
            if line[0] != '#':
                sys.exit('ERROR: wrong line!!!')
            pairs.append(line[1:-1])
            cols = fin.readline().split()
            X.append([float(_.split(':')[-1]) for _ in cols[1:]])
            y.append(cols[0])
        print(file=sys.stderr)
    return np.array(X), np.array(y), pairs
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
#rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
#rfc.fit(x_train, y_train)
#model="./model/rfc1"
#joblib.dump(rfc, model)
#rfc = joblib.load(model) 
#test_p = rfc.predict_proba(x_test)
####################################################################################

from sklearn.model_selection import KFold
def get_xdata(data,newx):
    datas=[]
    length=len(data)
    for i in range(length):
        sig_data=newx[data[i]]
        # print("1",data[i])
        # print("2",sig_data)
        datas.append(sig_data)
    datas=np.array(datas)
    return datas
def get_ydata(data,y):
    datas=[]
    length=len(data)
    for i in range(length):
        sig_data=y[data[i]]
        datas.append(sig_data)
    datas=np.array(datas)
    return datas 

def get_index(pairs,data,output_dir,test_p):
    for i in range(len(data)):
        index=data[i]
        score=test_p[i][1]
        with open(output_dir, 'a') as fout:
            print(pairs[index], score, file=fout)  

def addtwodimdict(thedict, key_a, key_b, val): 
    if key_a in thedict.keys():
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

import lightgbm as lgb
from xgboost import XGBClassifier
# if __name__ == '__main__':
def randf():
    myalloutput="./output/myrfcoutput"
    mytrain="./data/mynewtrain.txt"#86个特征
    x,y,pairs=generate_X_y(mytrain)
    # print("x:",x.shape)
    feature85 = []
    vec_dict = {}
    with open("./LE_pairs.txt", "r") as f:
        pairdatas = f.readlines()
        # print(data)
    for data in pairdatas:
        pairList = data.split(" ")
        addtwodimdict(vec_dict,pairList[0],pairList[1], pairList[2])
    for pair in pairs:
        pairList1 = pair.split(" ")
        score = vec_dict[pairList1[0]][pairList1[1]]
        feature85.append(float(score.rstrip("\n")))
        # file_handle.write(score)
        # for data in datas:
        #     pairList = data.split(" ")
        #     pair1 = pairList[0]+" "+pairList[1]
        #     if(pair == pair1):
        #         feature85.append(pairList[2])
        #         sum += 1
        #         file_handle.write(pairList[2])
        #         print(sum)
    a = np.expand_dims(np.array(feature85), axis=1)
    newx1=x[:,84:87] #deepfr and SSAfold
    # newx2=x[:,87:] ## ATRIPLR
    newx=np.hstack([newx1,a])
    # newx = newx1
    print("newx:",newx.shape)
    kf = KFold(n_splits=10)
    i=0
    for train, test in kf.split(newx):
        train_x=get_xdata(train,newx)
        train_y=get_ydata(train,y)
        test_x=get_xdata(test,newx)
        # test_y=get_ydata(test)
        # np.savetxt("./rfc/train"+str(i)+".txt", train_x)
        # np.savetxt("./rfc/trainy"+str(i)+".txt", train_y)
        # np.savetxt("./rfc/test"+str(i)+".txt", test_x)  
        rfc = lgb.LGBMClassifier(
                        num_leaves=1024,
                        is_unbalance = True,
                        learning_rate=0.05,
                        n_estimators=200)
        # rfc = RandomForestClassifier(
        #     n_estimators=200, n_jobs=-1,class_weight="balanced_subsample",
        #     min_samples_split=100, min_samples_leaf=50, max_features=4)
        # rfc = XGBClassifier(n_estimators=200, colsample_bytree = 0.8,
        #  scale_pos_weight = 70, max_depth = 10,
        #  eval_metric=['logloss','auc','error'], 
        #  use_label_encoder=False)
        rfc.fit(train_x,train_y)
        model_path="./rfcmodel/rfc"+str(i)
        # model="./model/rfc1"
        joblib.dump(rfc, model_path)
        # rfc = joblib.load(model_path) 
        test_p = rfc.predict_proba(test_x)
        get_index(pairs,test,myalloutput,test_p)
        # sig_out="./rfc/sig_out"+str(i)
        print (i)
        i=i+1

def randf2():
    mytrain="./data/mynewtrain.txt"#86个特征
    x,y,pairs=generate_X_y2(mytrain)
    # print("x:",x.shape)
    feature = []
    vec_dict = {}
    with open("LE_pairs.txt", "r") as f:
        pairdatas = f.readlines()
        # print(data)
    for data in pairdatas:
        pairList = data.split(" ")
        addtwodimdict(vec_dict,pairList[0],pairList[1], pairList[2])
    for i,pair in enumerate(pairs):
        pairList1 = pair.split(" ")
        score = vec_dict[pairList1[0]][pairList1[1]]
        feature.append(y[i]+" "+score.rstrip("\n"))
    file_handle=open("./imagdata2.txt",'a') 
    for score in feature:
        # print(score)
        file_handle.write(score+"\n")
    file_handle.close()

if __name__ == '__main__':
    randf2()