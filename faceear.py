#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import time
import tensorflow as tf
import numpy as np

from PIL import Image

from sklearn.decomposition import PCA

from nets.getembd_stnew import get_embd_st_face, get_embd_st_ear

from sklearn import metrics
from util.img_inputmt_stval import ImageReader
from util.input_arguments import arguments_st_eval

from sklearn.model_selection import KFold
from scipy import interpolate
from scipy.optimize import brentq

import math
import matplotlib.pyplot as plt
import copy
import sys

sys.path.append("/usr/local/MATLAB/R2017b/extern/engines/python/build/lib.linux-x86_64-2.7/")
import matlab.engine

eng = matlab.engine.start_matlab()

import matplotlib
from sklearn.linear_model import Lasso

import math


def ISRE(A, y, x):
    y = np.transpose([y])

    sce_list = []
    rootpath = '/home/tangxq/NDDR-CNN-net/datasets/er_train_pad/'
    list = os.listdir(rootpath)
    t = 0
    # x=d[0:553,:]
    # e_temp=d[553:,:]
    for i in range(0, len(list)):
        path = os.path.join(rootpath, list[i])
        list1 = os.listdir(path)
        dict_num = len(list1)
        x_temp = x[t:t + dict_num, :]

        '''A_temp =A[:,t:t+dict_num]          
        Ax = np.dot(A_temp,x_temp)
        isre = y-Ax
        sce = np.linalg.norm(isre, ord=2)'''
        sce = np.linalg.norm(x_temp, ord=1) / np.linalg.norm(x, ord=1)  # 稀疏贡献率

        # scee = np.linalg.norm(x_temp,ord=1)
        # cer=1.-math.exp(-6.*(scee/sce))

        sce_list.append(sce)
        t = t + dict_num
    return sce_list


def l1_ls(A, y):
    y = np.transpose([y])
    A = A.tolist()
    y = y.tolist()
    A = matlab.double(A)
    y = matlab.double(y)
    lambdaa = 0.05  # 可以改
    rel_tol = 0.01
    x = eng.l1_ls(A, y, lambdaa, rel_tol)
    x_sr = np.array(x)
    return x_sr


def cal_accuracy(y_score, y_true):
    c = zip(y_score, y_true)
    c = list(c)
    csort = sorted(c, key=lambda x: x[0])
    y_score, y_true = zip(*csort)
    y_score = list(y_score)
    y_true = list(y_true)
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    np.savetxt('/home/tangxq/NDDR-CNN-net/1022/news.txt', y_score, fmt='%f', delimiter=',')
    np.savetxt('/home/tangxq/NDDR-CNN-net/1022/newtrue1.txt', y_true, fmt='%f', delimiter=',')

    best_acc = 0
    best_th = 0
    frrlist = []
    farlist = []
    tprlist = []
    fprlist = []
    # choose_score=y_score[42000:]
    choose_score1 = y_score[:10000]
    choose_score2 = y_score[10000::10]
    choose_temp = np.append(choose_score1, choose_score2)
    for i in range(len(y_score)):
        # for i in range(len(choose_score)):
        th = y_score[i]

        y_test = (y_score >= th)  # 如果是稀疏贡献率改成  >=
        str = (y_test == y_true)
        tstr = (y_test & y_true)

        TP = np.sum(tstr)
        FN = np.sum(y_test < y_true)
        TP = float(TP)
        FN = float(FN)
        TPR = (TP / (TP + FN))

        FP = np.sum(y_test > y_true)
        TN = len(y_score) - TP - FN - FP
        FP = float(FP)
        TN = float(TN)
        FPR = (FP / (FP + TN))

        FAR = FPR
        FRR = 1 - TPR
        # tprlist.append( TPR)
        # fprlist.append(FPR)
        frrlist.append(FRR)
        farlist.append(FAR)

        acc = np.mean((y_test == y_true).astype(int))
        # print acc
        if acc > best_acc:
            best_acc = acc
            best_th = th
        acc = np.mean((y_test == y_true).astype(int))

    # 保存的文件改变
    np.savetxt('/home/tangxq/NDDR-CNN-net/1022/79er_scr_far.txt', farlist, fmt='%f', delimiter=',')

    np.savetxt('/home/tangxq/NDDR-CNN-net/1022/79er_scr_frr.txt', frrlist, fmt='%f', delimiter=',')

    v = list(map(lambda x: abs(x[0] - x[1]), zip(farlist, frrlist)))
    # auc_ear = metrics.auc(farlist, tpr_ear)
    minindex = v.index(min(v))
    eer = float(frrlist[minindex]) + float(farlist[minindex])
    ear = eer / 2

    return best_acc, best_th, ear


def eval():
    """Create the model and start the evaluation process."""

    embds1 = embds[0:553, :] #字典
    embds2 = embds[553:, :] #测试
    train_dict = train_dict[553:] # 图片名称
    truelable = truelable[0:553] # 字典的label

    Atrain_dict = embds1.transpose()
    Atest_dict = embds2.transpose()
    '''Atrain_dict=embds1
    Atest_dict=embds2'''
    fetrain_dict = {}
    fetest_dict = {}
    for i, each in enumerate(train_dict): #
        # fetrain_dict[each] = embds1[i]
        fetest_dict[each] = embds2[i]
    labellist = []
    for i in truelable:
        if i not in labellist:
            labellist.append(i)
    # print(labellist)
    x_sr = []
    labeltests = []
    isre_list = []
    isre_label = []
    all_isre_list = []
    all_isre_label = []
    testlabels = []
    pairs = train_dict
    m = []
    n = []
    k = 0
    distance = []
    rep = []
    totalcrf = []
    # print(fetest_dict.keys())
    for pair in pairs:
        id_y = fetest_dict[pair]
        splits = pair.split('/')
        label_test = int(splits[0])
        testlabels.append(label_test)

        # 稀疏表示
        x_sr = l1_ls(Atrain_dict, id_y)

        # x_sr=lasso_model(Atrain_dict,id_y)
        rep.append([])
        rep[k].append(x_sr)
        sce_list = ISRE(Atrain_dict, id_y, x_sr)
        totalAx = np.dot(Atrain_dict, x_sr)
        totalisre = id_y - totalAx
        totalsce = np.linalg.norm(totalisre, ord=2)
        totalcrf.append('[')
        totalcrf.append(totalsce)
        totalcrf.append(']')
        '''d_index=[]
        for index,nums in enumerate(truelable):
            if nums==label_test:
                d_index.append(index)
        c =copy.deepcopy(truelable)       
        for q in range(0,len(c)):
            c[q]=int(0)
        for ab in d_index:
            c[ab]=int(1)'''

        d_index = labellist.index(label_test)
        c = copy.deepcopy(labellist)
        for q in range(0, len(c)):
            c[q] = int(0)
        c[d_index] = int(1)
        all_isre_list = all_isre_list + sce_list
        all_isre_label = all_isre_label + c
        m.append([])
        m[k].append(sce_list)
        n.append([])
        n[k].append(c)
        k = k + 1
    print(k)
    fl = open('/home/tangxq/NDDR-CNN-net/1022/79er_scr_pipeilist.txt', 'w')
    fd = open('/home/tangxq/NDDR-CNN-net/1022/79er_scr_pipeilabel.txt', 'w')
    fs = open('/home/tangxq/NDDR-CNN-net/1022/79er_scr_CRF.txt', 'w')
    fl.write(str(m))
    fd.write(str(n))
    fs.write(str(totalcrf))
    acc, th, eer = cal_accuracy(all_isre_list, all_isre_label)
    # print(testlabels)

    print('acc:%', acc)
    print('th:%', th)
    print('eer:%', eer)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    eval()
