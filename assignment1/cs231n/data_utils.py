# -*- coding: utf-8 -*-
"""
Created on 2017/4/20 19:55

@author: luoxiang
"""

import pickle
import numpy as np
import os
def load_CIFAR_batch(filename):
    '''load single batch cifar'''
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        X = data[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')     #shape(10000,32,32,3),但并不等同于直接reshape(10000,32,32,3),因为数据的划分会不一样
        Y = np.array(data[b'labels'])                                                   #可能是因为版本的问题，必须加上b二进制才能取这跟key
        return X,Y
def load_CIFAR10(path):
    '''
    一次载入所有数据
    Args:
        path: ./cifar10
    Returns:
        Xtr: shape of (50000,32,32,3)
        Ytr: shape of (50000,)
        Xte: shape of (10000,32,32,3)
        Yte: shape of (10000,)
    '''
    xs = []
    ys = []
    for i in range(1,6):
        batchpath = os.path.join(path,'data_batch_%d'%i)
        X, Y = load_CIFAR_batch(batchpath)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)    #将numpy array进行合并，默认纵向合并(axis=0)
    Ytr = np.concatenate(ys)
    del X,Y
    Xte, Yte = load_CIFAR_batch(os.path.join(path,'test_batch'))
    return Xtr, Ytr, Xte, Yte