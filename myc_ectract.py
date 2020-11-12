#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :myc_extract.py
@说明    :
@时间    :2020/11/04 15:58:30
@作者    :Mayc
@版本    :0.1
'''
import pdb
from config import get_config
import argparse
from myc_Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans

conf = get_config(training=False)
learner = face_learner(conf, inference=True)
learner.load_state(conf, 'final.pth', model_only=True, from_save_folder=True)
import numpy as np
import cv2 as cv
import torch
import os


def load_dataset(root_dir):
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
    list_all = os.listdir(root_dir)
    for i in range(0, len(list_all)):
        # path: try_test/1
        path = os.path.join(root_dir, list_all[i])
        folder_image_name = os.listdir(path)
        for image_name in (folder_image_name):
            label = image_name.split('-')[0]
            print("image_name, i:", image_name, label)
            data_array = cv.imread(root_dir + '/' + label + '/' + image_name)
            # print("array_shape:", np.shape(data_array))
            data.append(data_array)
            labels.append(label)
    data = np.array(data)
    print("shape:", np.shape(data))
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    return data, labels


def extraction(root_dir):
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
    array, label = load_dataset(root_dir)
    data = learner.extract(conf, array)
    return data, label

    # data = learner.extract(conf, data_array)


if __name__ == "__main__":
    extraction('gt_train_pad')