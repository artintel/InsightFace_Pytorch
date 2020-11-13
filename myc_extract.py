# -*- encoding: utf-8 -*-
'''
@文件    :myc_extract.py
@说明    :
@时间    :2020/11/13 16:23:30
@作者    :Mayc
@版本    :1.0
'''
from config import get_config
from myc_Learner import face_learner
conf = get_config(training=False)
learner = face_learner(conf, inference=True)
learner.load_state(conf, 'final.pth', model_only=True, from_save_folder=True)
import numpy as np
import cv2 as cv
import torch
import copy
import os
import sys
sys.path.append("/usr/local/MATLAB/R2017b/extern/engines/python/build/lib.linux-x86_64-2.7/")
sys.__egginsert = len(sys.path)
import matlab.engine
eng = matlab.engine.start_matlab()


def cal_accuracy(y_score, y_true):
    c = zip(y_score, y_true)
    c = list(c)
    csort = sorted(c, key=lambda x: x[0])
    y_score, y_true = zip(*csort)
    y_score = list(y_score)
    y_true = list(y_true)
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    np.savetxt('news.txt', y_score, fmt='%f', delimiter=',')
    np.savetxt('newtrue1.txt', y_true, fmt='%f', delimiter=',')

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

        y_test = (y_score <= th)  # 如果是稀疏贡献率改成  >=
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
    np.savetxt('79er_isre_far.txt', farlist, fmt='%f', delimiter=',')

    np.savetxt('79er_isre_frr.txt', frrlist, fmt='%f', delimiter=',')

    v = list(map(lambda x: abs(x[0] - x[1]), zip(farlist, frrlist)))
    # auc_ear = metrics.auc(farlist, tpr_ear)
    minindex = v.index(min(v))
    eer = float(frrlist[minindex]) + float(farlist[minindex])
    ear = eer / 2

    return best_acc, best_th, ear

def l1_ls(A,y):
    y = np.transpose([y])
    A =A.tolist()
    y =y.tolist()
    A = matlab.double(A)
    y = matlab.double(y)
    lambdaa = 0.05   # 可以改
    rel_tol = 0.01
    x = eng.l1_ls(A, y, lambdaa, rel_tol)
    x_sr = np.array(x)
    return x_sr


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

        A_temp = A[:, t:t + dict_num]
        Ax = np.dot(A_temp, x_temp)
        isre = y - Ax
        sce = np.linalg.norm(isre, ord=2)
        # sce = np.linalg.norm(x_temp,ord=1)/np.linalg.norm(x,ord=1)  #稀疏贡献率

        # scee = np.linalg.norm(x_temp,ord=1)
        # cer=1.-math.exp(-6.*(scee/sce))

        sce_list.append(sce)
        t = t + dict_num
    return sce_list

def load_dataset(root_dir1, root_dir2):
    """
    @description :
    @param :
        root_dir - 数据集的路径。格式为
            root_dir/
                - 1/
                    - 1-1.jpg
                    - 1-2.jpg
                    ...
                - 2
                - 3
                ...
                - len
    @相关元素 :
        path: file_name/1
        image_name: ['1-15.jpg', '1-11.jpg', '1-3.jpg', '1-14.jpg', '1-9.jpg', '1-1.jpg', '1-5.jpg']
                    ['2-6.jpg', '2-8.jpg', '2-9.jpg', '2-2.jpg', '2-4.jpg', '2-3.jpg', '2-12.jpg'

    @Returns : 返回加载好的torch数据
    """

    data = []
    labels = []
    data_dict = []
    root = []
    root.append(root_dir1)
    root.append(root_dir2)
    for k in range(2):
        list_all = os.listdir(root[k])
        for i in range(0, len(list_all)):
            # path: try_test/1
            path = os.path.join(root[k], list_all[i])
            folder_image_name = os.listdir(path)
            for image_name in (folder_image_name):
                label = image_name.split('-')[0]
                # print("image_name, i:", image_name, label)
                data_array = cv.imread(root[k] + '/' + label + '/' + image_name)
                data_dict.append(label + '/' + image_name)
                # print("array_shape:", np.shape(data_array))
                data.append(data_array)
                labels.append(label)


    data = np.array(data)
    labels = np.array(labels)
    data_dict = np.array(data_dict)
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    return data, labels, data_dict


def extraction(root_dir1, root_dir2):
    """
    @description  :  把通过load_datase穿入的数据进行extrac提取操作
    @param  :
        root_dir - 数据集的路径。格式为
        root_dir/
            - 1/
                - 1-1.jpg
                - 1-2.jpg
                ...
            - 2
            - 3
            ...
            - len
    @Returns  : 返回 (n, 512) 的特征array
    """
    array, label, data_dict = load_dataset(root_dir1, root_dir2)
    data = learner.extract(conf, array)
    # print(np.shape(data))
    return data, label, data_dict

def eval(root_dir1, root_dir2):
    embds1 = []
    embds2 = []
    embds = []
    testlabel = []
    train_dict = []
    embds, lable, img_dict = extraction(root_dir1, root_dir2)
    embds1 = embds[0:350] # 字典特征
    embds2 = embds[350:] # 测试特征
    test_dict = img_dict[350:] # 测试集文件名
    trainlable = lable[0:350] # 字典集lable
    embds1 = embds1/np.linalg.norm(embds1, axis=1, keepdims=True)
    embds2 = embds2/np.linalg.norm(embds2, axis=1, keepdims=True)
    Atrain_dict = embds1.transpose()
    Atest_dict = embds2.transpose()
    # print("Atrain_dict: {}, Atest_dict: {}".format(np.shape(Atrain_dict), np.shape(Atest_dict)))
    fetrain_dict={}
    fetest_dict={}
    for i, each in enumerate(test_dict):
        fetest_dict[each] = embds2[i]
    labellist=[]
    for i in trainlable:
        if i not in labellist:
            labellist.append(i)
    # print(labellist)
    x_sr=[]
    labeltests=[]
    isre_list=[]
    isre_label=[]
    all_isre_list=[]
    all_isre_label=[]
    testlabels=[]
    pairs=test_dict
    m=[]
    n=[]
    k = 0
    distance=[]
    rep=[]
    totalcrf=[]
    # print(fetest_dict.keys())
    for pair in pairs:
        id_y = fetest_dict[pair]
        splits = pair.split('/')
        label_test = int(splits[0])
        testlabels.append(label_test)

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

        labellist = list(map(int, labellist))
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
    fl = open('79er_isre_pipeilist.txt', 'w')
    fd = open('79er_isre_pipeilabel.txt', 'w')
    fs = open('79er_isre_CRF.txt', 'w')
    fl.write(str(m))
    fd.write(str(n))
    fs.write(str(totalcrf))
    print(all_isre_label)
    print(all_isre_list)
    acc, th, eer = cal_accuracy(all_isre_list, all_isre_label)
    # print(testlabels)

    print('acc:%', acc)
    print('th:%', th)
    print('eer:%', eer)



if __name__ == "__main__":
    eval('gt_train_pad', 'gt_testt_pad')