# -*- coding: utf-8 -*-
"""
Created on 2017/5/10 14:01

@author: luoxiang
"""

from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):
    def __init__(self):
        self.W = None
    def train(self,X,y,learning_rate=1e-3,reg=1e-5,num_iters=100,batch_size=200,verbose=False):
        num_train,dim = X.shape
        num_class = np.max(y) + 1 # start from 0
        if self.W is None: # 初始化权值
            self.W = np.random.randn(dim,num_class) * 0.001
        loss_history = []
        for it in range(num_iters):
            #默认每次随机取200个作为minibatch,称为MGD
            #choice第一个参数为整数或者序列，第二个参数为随机取样个数，replace为False时表示无放回
            #采样，还可以传入p为每个样本的采样概率
            sample_index = np.random.choice(num_train,batch_size,replace=False)
            x_batch = X[sample_index,:]
            y_batch = y[sample_index]
            loss, grad = self.loss(x_batch,y_batch,reg)
            #每一次都保存loss，利于后面可视化loss的过程
            loss_history.append(loss)
            self.W += -learning_rate * grad
            if verbose and it % 100 ==0:
                print('Iteration %d/%d: loss %f'%(it,num_iters,loss))
        return loss_history
    def predict(self,X):
        scores = np.dot(X,self.W)
        y_pred = np.argmax(scores,axis=1)       #得分最高的认为是正确分类
        return y_pred
    def loss(self,X_batch,y_batch,reg):
        # this will be overwrite in subclass
        pass
class LinearSVM(LinearClassifier):
    #继承线形分类器的所有方法
    def loss(self,X_batch,y_batch,reg):
        return svm_loss_vectorized(self.W,X_batch,y_batch,reg)
class Softmax(LinearClassifier):
    def loss(self,X_batch,y_batch,reg):
        return softmax_loss_vectorized(self.W,X_batch,y_batch,reg)