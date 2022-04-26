# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:26:03 2021

@author: 75955
"""

from __future__ import print_function, division
from PIL import Image  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import torch.nn as nn
from torchvision.transforms import transforms as T
import pandas as pd
#from utils import train, resnet
import PIL.Image as Image
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
import math
import glob
import scipy.spatial.distance as distance
import csv
import codecs
# from residual_attention_network import ResidualAttentionModel_448input as ResidualAttentionModel


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = ResidualAttentionModel().to(device)
# model.load_state_dict(torch.load("/data/liuyan/code/two_supervised/hankeattentionfold/VGG165.pt",map_location={'cuda:1':'cuda:0'}))
# model.eval()
import sys
import numpy as np

def parse_listfile(listfile, col_list=None):
    lines = []
    try:
        with open(listfile, 'r') as fin:
            if col_list:
                for line in fin:
                    cols = line.split()
                    lines.append(tuple(cols[i-1] for i in col_list))
            else:
                for line in fin:
                    lines.append(tuple(line.split()))
    except Exception as e:
        print('ERROR: wrong list file "%s"' % listfile, e, file=sys.stderr)
    return lines

def parse_pairwise_score(pairwise_score):
    lines = parse_listfile(pairwise_score, [1, 2, 3])
    pairs, scores = [], []
    for i in lines:
        pairs.append((i[0], i[1]))
        scores.append(float(i[2]))
    return pairs, scores

def parse_pair(pairlist):
    return parse_listfile(pairlist, [1, 2])

def get_top_indexs(scores):
    scores = np.array(scores)
    top_inds = np.argsort(scores)[::-1]
    # get top1
    top1 = list(top_inds[:1])
    for i in top_inds[1:]:
        if scores[i] == scores[top1[0]]:
            top1.append(i)
    # get top5
    top5 = list(top_inds[:5])
    for i in top_inds[5:]:
        if scores[i] == scores[top5[4]]:
            top5.append(i)
    assert len(top1) < 15 and len(top5) < 30
    return top1, top5

def calcute_fold_top1_and_top5(lindahl_pairwise_score, lindahl_data):
    # score dict
    pairs, scores = parse_pairwise_score(lindahl_pairwise_score)
    score_dict = {}
    for i, p in enumerate(pairs):
        if p[0] not in score_dict:
            score_dict[p[0]] = []
        score_dict[p[0]].append((p, scores[i]))

    # lindahl data
    lindahl_pairs = parse_pair(lindahl_data)
    lindahl_names = []
    for i in lindahl_pairs:
        if i[0] not in lindahl_names:
            lindahl_names.append(i[0])
    #lindahl_names = list(set([i[0] for i in lindahl_pairs]))
    
    # calculte top1 top5
    top = [0, 0]
    for i in lindahl_names:
        tmp_scores = [s[1] for s in score_dict[i]]
        top1, top5 = get_top_indexs(tmp_scores)
        for k in top1:
            if score_dict[i][k][0] in lindahl_pairs:
                top[0] += 1
                break
        for k in top5:
            if score_dict[i][k][0] in lindahl_pairs:
                top[1] += 1
                break
    print('Test_number:', len(lindahl_names))
    print('Top_number:', top)
    print('Sensitivity:', '%4.1f %4.1f' % tuple([i/len(lindahl_names)*100 for i in top]))
    Sensitivity = '%4.1f %4.1f' % tuple([i/len(lindahl_names)*100 for i in top])
    with open("./result.txt","a+") as f:
        f.writelines('Sensitivity:' + str(Sensitivity)+ '\n')
    return Sensitivity[0]
        

def read_contact_map(contact_logist_path):
    contact_logist=np.loadtxt(contact_logist_path)
    if np.shape(contact_logist)[0]<256:
        contact_logist=Supplementary_matrix(contact_logist,256)
    else:
        contact_logist=crop_matrix(contact_logist,256)
    return contact_logist
def Supplementary_matrix(matrix,size):
    len_row,len_col=np.shape(matrix)
    new_matrix=np.zeros((size,size))
    for i in range(len_row):
        for j in range(len_col):
            new_matrix[i][j]=matrix[i][j]
    return new_matrix
#crop the matrix so that the size is fixed
def crop_matrix(matrix,size):
    crop_matrix=matrix[0:size,0:size]
    return crop_matrix
def read_samplecontact(contact_path):
    image=Image.open(contact_path).convert('L')
    image=np.array(image)
    return image
# def eval_feature(model,contact_logist_path):
#     contact_logist=read_samplecontact(contact_logist_path)
# #    print (contact_logist.size())
#     contact_logists=torch.tensor(np.array(contact_logist)).unsqueeze(0).unsqueeze(1).float().to(device)
#     features = []
#     def hook(module, input, output):
#         features.append(output.clone().detach())
#     handle = model.fc[0].register_forward_hook(hook)
#     y = model(contact_logists)
#     handle.remove()
# #    print ()
#     return features[0]
def get_LE_label_index(filename):
    LE_label_index={}
    LE_labels=[]
    names=[]
    with open(filename,'r') as fin:
        for line in fin:
            line=line.rstrip('\n')
            index=line.find(" ")
            name=line[0:index]
            label=line[index+1:]
            LE_label_index[name]=label
            names.append(name)
            LE_labels.append(label)
    return LE_label_index, names, LE_labels
# def get_average_feature(model,image_rootdir):
#     sum_feature=np.zeros((1,2048))
#     # image_rootdir="./LEsample_contactmap/1aab-d1aab/"
#     path_file_numbers=glob.glob(image_rootdir+"*.jpg")
#     for i in range(len(path_file_numbers)):
#         jpg_path=path_file_numbers[i]
#         # print (jpg_path)
#         sing_feature=eval_feature(model,jpg_path).cpu().numpy()
#         sum_feature=sum_feature+sing_feature
#     final_feature=sum_feature/len(path_file_numbers)
#     return final_feature
def calculate_similarity(feature1, feature2):
    return 1 - distance.cosine(feature1, feature2)
def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))
def calculate_eusimilarity(feature1, feature2):
    return 1 - eucliDist(feature1, feature2)

from scipy.stats import pearsonr    
def calculate_pccsimilarity(feature1, feature2):
    return pearsonr(feature1, feature2)[0]
# model.eval()
# import os
# LE_dict={}
# LE_label_index,names,LE_labels=get_LE_label_index("lindahl_name_label_list")

# for i in range(len(names)):
#     print (i)
        
#     jpgs_path="LEsample_contactmap/"+names[i]+"/"
#     single_feature=get_average_feature(model,jpgs_path)
#     LE_dict.update({names[i]:single_feature})

#np.save('RANfold_LE_dict.npy', LE_dict)
# import pickle
# list3_file=open("RANFOLD_feature.pickle","wb")
# pickle.dump(LE_dict,list3_file)
# list3_file.close()
# file_handle=open("LE_pairs.txt",mode='a')
# for i in range(len(names)):
#     query=LE_dict[names[i]].reshape(2048)
        
#     for j in range(len(names)):
#         template=LE_dict[names[j]].reshape(2048)
#         if i!=j:
#             score=calculate_similarity(query,template)
#             file_handle.write(names[i]+" "+names[j]+" "+str(score)+"\n")
# file_handle.close()
def result():
    os.system('cd ./')
    os.system('pwd')
    os.system('grep -F -v -f  ./lindahl_family ./output/myrfcoutput > LE_pairs-family')
    os.system('grep -F -v -f  ./lindahl_superfamily LE_pairs-family > LE_pairs-family-super')

    lindahl_pairwise_score="./output/myrfcoutput"
    lindahl_superfamily="./lindahl_superfamily"
    lindahl_family="./lindahl_family"

    lindahl_superfamilypairwise_score="LE_pairs-family"

    lindahl_foldpairwise_score="LE_pairs-family-super"
    lindahl_fold="./lindahl_fold"
    Sensitivity = calcute_fold_top1_and_top5(lindahl_foldpairwise_score, lindahl_fold)
    calcute_fold_top1_and_top5(lindahl_superfamilypairwise_score,lindahl_superfamily)
    calcute_fold_top1_and_top5(lindahl_pairwise_score, lindahl_family)
        
    os.system('rm LE_pairs-family')
    os.system("rm LE_pairs-family-super")
    os.system("rm LE_pairs.txt")
    os.system("rm ./output/myrfcoutput")
    print ("delete LE_pairs")
    return Sensitivity

def result2():
    os.system('cd ./')
    os.system('pwd')
    os.system('grep -F -v -f  ./lindahl_family LE_pairs.txt > LE_pairs-family')
    os.system('grep -F -v -f  ./lindahl_superfamily LE_pairs-family > LE_pairs-family-super')

    lindahl_pairwise_score="LE_pairs.txt"
    lindahl_superfamily="./lindahl_superfamily"
    lindahl_family="./lindahl_family"

    lindahl_superfamilypairwise_score="LE_pairs-family"

    lindahl_foldpairwise_score="LE_pairs-family-super"
    lindahl_fold="./lindahl_fold"
    Sensitivity = calcute_fold_top1_and_top5(lindahl_foldpairwise_score, lindahl_fold)
    calcute_fold_top1_and_top5(lindahl_superfamilypairwise_score,lindahl_superfamily)
    calcute_fold_top1_and_top5(lindahl_pairwise_score, lindahl_family)
        
    os.system('rm LE_pairs-family')
    os.system("rm LE_pairs-family-super")
    os.system("rm LE_pairs.txt")
    # os.system("rm ./output/myrfcoutput")
    print ("delete LE_pairs")
    return Sensitivity

if __name__ == '__main__':
    result2()
################################################
# file_handle=open("LE_pairs.txt",mode='a')
# for i in range(len(names)):
#     query=LE_dict[names[i]].reshape(2048)
        
#     for j in range(len(names)):
#         template=LE_dict[names[j]].reshape(2048)
#         if i!=j:
#             score=calculate_eusimilarity(query,template)
#             file_handle.write(names[i]+" "+names[j]+" "+str(score)+"\n")
# file_handle.close()
# os.system('cd ./hanke')
# os.system('pwd')
# os.system('grep -F -v -f  ./hanke/lindahl_family LE_pairs.txt > LE_pairs-family')
# os.system('grep -F -v -f  ./hanke/lindahl_superfamily LE_pairs-family > LE_pairs-family-super')

# lindahl_pairwise_score="LE_pairs.txt"
# lindahl_superfamily="./hanke/lindahl_superfamily"
# lindahl_family="./hanke/lindahl_family"

# lindahl_superfamilypairwise_score="LE_pairs-family"

# lindahl_foldpairwise_score="LE_pairs-family-super"
# lindahl_fold="./hanke/lindahl_fold"
# calcute_fold_top1_and_top5(lindahl_foldpairwise_score, lindahl_fold)
# calcute_fold_top1_and_top5(lindahl_superfamilypairwise_score,lindahl_superfamily)
# calcute_fold_top1_and_top5(lindahl_pairwise_score, lindahl_family)
# os.system('rm LE_pairs-family')
# os.system("rm LE_pairs-family-super")
# os.system("rm LE_pairs.txt")
# print ("delete LE_pairs")    
    # os.system('rm LE_pairs-family')
    # os.system("rm LE_pairs-family-super")
    # os.system("rm LE_pairs.txt")
    # print ("delete LE_pairs")
# file_handle=open("LE_pairs.txt",mode='a')
# for i in range(len(names)):
#     query=LE_dict[names[i]].reshape(2048)
        
#     for j in range(len(names)):
#         template=LE_dict[names[j]].reshape(2048)
#         if i!=j:
#             score=calculate_pccsimilarity(query,template)
#             file_handle.write(names[i]+" "+names[j]+" "+str(score)+"\n")
# file_handle.close()
# os.system('cd ./hanke')
# os.system('pwd')
# os.system('grep -F -v -f  ./hanke/lindahl_family LE_pairs.txt > LE_pairs-family')
# os.system('grep -F -v -f  ./hanke/lindahl_superfamily LE_pairs-family > LE_pairs-family-super')

# lindahl_pairwise_score="LE_pairs.txt"
# lindahl_superfamily="./hanke/lindahl_superfamily"
# lindahl_family="./hanke/lindahl_family"

# lindahl_superfamilypairwise_score="LE_pairs-family"

# lindahl_foldpairwise_score="LE_pairs-family-super"
# lindahl_fold="./hanke/lindahl_fold"
# calcute_fold_top1_and_top5(lindahl_foldpairwise_score, lindahl_fold)
# calcute_fold_top1_and_top5(lindahl_superfamilypairwise_score,lindahl_superfamily)
# calcute_fold_top1_and_top5(lindahl_pairwise_score, lindahl_family)
# os.system('rm LE_pairs-family')
# os.system("rm LE_pairs-family-super")
# os.system("rm LE_pairs.txt")
# print ("delete LE_pairs")  