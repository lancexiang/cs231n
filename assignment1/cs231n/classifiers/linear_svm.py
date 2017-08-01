# -*- coding: utf-8 -*-
"""
Created on 2017/4/22 11:08

@author: luoxiang
"""

import numpy as np

def svm_loss_naive(W,X,y,reg=1e-5):
    '''
    Structed SVM loss function, naive implementation
    Args:
        W: A numpy array of shape(D,C) cotaining weight
        X: A numpy array of shape(N,D) cotaining train data
        y: A numpy array of shape(N,) cataining training lebels
        reg:(float)Regularization stength
    Returns:
        loss of a single float
    '''
    dw = np.zeros_like(W)
    loss = 0.0
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = scores[j] - scores[y[i]] + 1
            if margin>0:
                loss += margin
                #loss = Σmax(0,WXi - WXyi + 1),所以有以下的求导公式，也就是梯度计算公式
                dw[:,y[i]] += -X[i]
                dw[:,j] += X[i]
    loss /= num_train
    dw = dw/num_train


    loss += 0.5 * reg * np.sum(W*W)                                   #L2距离
    dw += reg * W

    return loss,dw
def svm_loss_vectorized(W,X,y,reg=1e-5):
    '''
    Structed SVM loss function, vectorized implementation(vectorized means no loops)
    Args:
        W: A numpy array of shape(D,C) cotaining weight
        X: A numpy array of shape(N,D) cotaining train data
        y: A numpy array of shape(N,) cataining training lebels
        reg:(float)Regularization stength

    Returns:
        loss of a single float
    '''
    dw = np.zeros_like(W)
    loss = 0.0
    num_train = X.shape[0]
    scores = X.dot(W)              #shape(N,C)
    #用真实类标y确定本应该分数最大的列索引位置位置，取玩maximum后会变成行向量，需要增加一维，防止出错
    margin = scores - scores[np.arange(num_train),y][:,np.newaxis]+1
    margin = np.maximum(0, margin)
    margin[np.arange(num_train),y] = 0        #正确类标不需要与自己作比较
    loss += np.sum(margin)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)     #plus the regularizatin using L2 distace of w,which is also name euclidean metric
    margin[margin > 0] = 1.0
    row_sum = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] = -row_sum
    dw += np.dot(X.T,margin)/num_train + reg * W
    return loss,dw