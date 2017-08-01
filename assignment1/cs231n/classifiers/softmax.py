# -*- coding: utf-8 -*-
"""
Created on 2017/4/22 14:29

@author: luoxiang
"""

import numpy as np

def softmax_loss_naive(W,X,y,reg):
    '''
    最原始的一种实现，使用循环实现，并求得梯度
    Args:
            W: An numpy array of shape(D,C) cotaining weights
            X: An numpy array of shape(N,D) cotaining train data
            y: An numpy array of shape(N,) cotaining train data labels
            reg: (float)Regularization strength
    Returns:
        loss of a single float
        gradient with respect to W of the same shape as W
    '''
    dw = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]
    loss=0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        #防止数据过大，进行指数运算时溢出
        scores -= np.max(scores)
        prob = np.exp(scores) / np.sum(np.exp(scores))
        for j in range(num_class):
            if j==y[i]:
                loss += -np.log(prob[j])
                dw[:,j] += (prob[j]-1) * X[i]                  #可以由损失函数推导而来，过程稍微复杂，需用复合求导进行化简
    # 损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
    #http://upload-images.jianshu.io/upload_images/2301760-1c7b8c12bbe6a1bc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240
            else:
                dw[:,j] += prob[j] * X[i]
    # 梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
    loss /= num_train
    loss += 0.5*reg * np.sum(W*W)                               #此处多个0.5时为了与求导后的2此项系数消去
    dw = dw/num_train + reg * W
    return loss,dw
def softmax_loss_vectorized(W,X,y,reg):
    '''
    Sofxmax loss function, vectorized implementation
    Args:
        W: An numpy array of shape(D,C) cotaining weights
        X: An numpy array of shape(N,D) cotaining train data
        y: An numpy array of shape(N,) cotaining train data labels
        reg: (float)Regularization strength
    Returns:
        loss of a single float
        gradient with respect to W of the same shape as W
    '''
    num_train = X.shape[0]
    dw = np.zeros_like(W)
    loss =0.0
    scores = X.dot(W)
    scores -= np.max(scores)                                #如果没有这步操作，可能会因为数值很大而出现指数爆炸的问题
    probs = np.exp(scores)/(np.sum(np.exp(scores),axis=1,keepdims=True))   #同样的sum之后会编成一维向量，需增加一维变为数组，否则会按行广播而出现除法不是每一行除的是那行的加和
    loss += np.sum(-np.log(probs[np.arange(num_train),y]))          #交叉熵损失计算规则
    keepprobs = np.zeros_like(probs)
    keepprobs[np.arange(num_train),y] = 1.0
    dw = -np.dot(X.T,keepprobs - probs)/num_train + reg * W
    loss /= num_train
    loss += 0.5*reg * np.sum(W * W)
    return loss,dw