#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from nets.getembd_stnew import get_embd_st_face, get_embd_st_ear
from util.img_inputmt_stval import ImageReader
from util.input_arguments import arguments_st_eval
import copy
import sys

sys.path.append("/usr/local/MATLAB/R2017b/extern/engines/python/build/lib.linux-x86_64-2.7/")
import matlab.engine

eng = matlab.engine.start_matlab()
from sklearn.linear_model import Lasso

slim = tf.contrib.slim

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def ISRE(A, y, x):
    y = np.transpose([y])

    sce_list = []
    rootpath = 'er_train_pad/'
    list = os.listdir(rootpath)
    t = 0
    # x=d[0:553,:]
    # e_temp=d[553:,:]
    for i in range(0, len(list)):
        path = os.path.join(rootpath, list[i])
        list1 = os.listdir(path)
        dict_num = len(list1)
        x_temp = x[t:t + dict_num, :]
        # sce = np.linalg.norm(x_temp, ord=1) / np.linalg.norm(x, ord=1)  # 稀疏贡献率

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


def lasso_model(A, y):
    y = np.transpose([y])
    model = Lasso(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=100000, normalize=False, positive=False,
                  precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
    model.fit(A, y)
    c = (model.coef_)
    c = c.reshape(-1, 1)
    return c


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
    np.savetxt('50er_scr_far.txt', farlist, fmt='%f', delimiter=',')

    np.savetxt('50er_scr_frr.txt', frrlist, fmt='%f', delimiter=',')

    v = list(map(lambda x: abs(x[0] - x[1]), zip(farlist, frrlist)))
    # auc_ear = metrics.auc(farlist, tpr_ear)
    minindex = v.index(min(v))
    eer = float(frrlist[minindex]) + float(farlist[minindex])
    ear = eer / 2
    return best_acc, best_th, ear


def eval():
    """Create the model and start the evaluation process."""
    args = arguments_st_eval()

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Encode data.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_list,
            args.data_list_1,
            input_size=None,
            coord=coord)
        image_1, label_1, image_list = reader.dequeue(args.batch_size)
    # Create network.
    if args.network == 'resnet_v2_50':
        net1, end_points_1 = get_embd_st_face(image_1, is_training_bn=False, is_training_dropout=False, reuse=None)

    restore_var = tf.global_variables()

    ear_output = net1

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    if args.restore_from is not None:
        if tf.gfile.IsDirectory(args.restore_from):
            folder_name = args.restore_from
            checkpoint_path = tf.train.latest_checkpoint(args.restore_from)
        else:
            folder_name = args.restore_from.replace(args.restore_from.split('/')[-1], '')
            checkpoint_path = args.restore_from

        tf.train.Saver(var_list=restore_var).restore(sess, checkpoint_path)
        print("Restored model parameters from {}".format(checkpoint_path))

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    embds1 = []
    embds2 = []
    embdss = []
    truelable = []
    train_dict = []

    for step in range(args.num_steps):
        # tf.get_variable_scope().reuse_variables()
        imgs_1, ear_out_1, label, img_list = sess.run([image_1, ear_output, label_1, image_list])

        embdss += list(ear_out_1)
        truelable += list(label)
        train_dict += list(img_list)

    train_dict = [i[48:] for i in train_dict]
    embds = np.array(embdss)

    embds1 = embds[0:350, :]
    embds2 = embds[350:, :]
    train_dict = train_dict[350:]
    truelable = truelable[0:350]

    embds1 = embds1 / np.linalg.norm(embds1, axis=1, keepdims=True)
    embds2 = embds2 / np.linalg.norm(embds2, axis=1, keepdims=True)

    truelable = [int(i) for i in truelable]
    print('embds1:%', embds1.shape)
    print('embds2:%', embds2.shape)

    print(truelable)

    print(train_dict)
    Atrain_dict = embds1.transpose()
    Atest_dict = embds2.transpose()
    '''Atrain_dict=embds1
    Atest_dict=embds2'''
    fetrain_dict = {}
    fetest_dict = {}
    for i, each in enumerate(train_dict):
        # fetrain_dict[each] = embds1[i]
        fetest_dict[each] = embds2[i]
    labellist = []
    for i in truelable:
        if i not in labellist:
            labellist.append(i)
    # print(labellist)
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
    for pair in pairs:
        id_y = fetest_dict[pair]
        splits = pair.split('/')
        label_test = int(splits[0])
        testlabels.append(label_test)

        x_sr = l1_ls(Atrain_dict, id_y)
        rep.append([])
        rep[k].append(x_sr)
        sce_list = ISRE(Atrain_dict, id_y, x_sr)
        totalAx = np.dot(Atrain_dict, x_sr)
        totalisre = id_y - totalAx
        totalsce = np.linalg.norm(totalisre, ord=2)
        totalcrf.append('[')
        totalcrf.append(totalsce)
        totalcrf.append(']')

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
    fl = open('50er_scr_pipeilist.txt', 'w')
    fd = open('50er_scr_pipeilabel.txt', 'w')
    fs = open('50er_scr_CRF.txt', 'w')
    fl.write(str(m))
    fd.write(str(n))
    fs.write(str(totalcrf))
    acc, th, eer = cal_accuracy(all_isre_list, all_isre_label)

    print('acc:%', acc)
    print('th:%', th)
    print('eer:%', eer)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    eval()
