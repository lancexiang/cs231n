# -*- coding: utf-8 -*-
"""
Created on 2017/4/20 20:41

@author: luoxiang
"""

import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,Y):
        '''
        It's just remember the train data and label
        Args:
            X:An numpy array of shape(50000,3072) containing all train data
            Y:An numpy array of shape(50000,) contain label of train data
        '''
        self.Xtr = X
        self.Ytr = Y
    def predict(self, X , k=1, num_loop=0):
        if num_loop == 0:
            dists = self.compute_distances_no_loops(X)
        if num_loop == 1:
            dists = self.compute_distances_one_loop(X)
        if num_loop == 2:
            dists = self.compute_distances_two_loops(X)
        return self.predict_labels(dists,k=k)
    def compute_distances_two_loops(self, X):
        '''
        Referencesompute destance between two neighbors
        Here the distance metric is L2 distance,i.e. Euclidean distance
        Args:
            X:An numpy array of the shape(num_test,num_train) containing test data
        Returns:
        '''
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test,num_train))
        #两层循环可以逐个更新
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.sum(np.square(self.Xtr[j,:] - X[i,:])))
        return dists
    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test,num_train))
        #一层循环可以逐行更新
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.Xtr-X[i,:]),axis=1))
        return dists
    def compute_distances_no_loops(self, X):
        #此处需要一点推导过程，原来是两个数差的平方的求和，写成矢量化没有循环的话
        #须将差的放平展开才能矢量化的对数据进行正确的操作
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        # dists = np.zeros((num_test,num_train))
        dists = np.multiply(np.dot(X,self.Xtr.T),-2)     #-2*A*B
        sq1 = np.sum(np.square(X),axis=1,keepdims=True)  #A^2
        sq2 = np.sum(np.square(self.Xtr),axis=1)         #B^2,主意根据dists的形状来判断要不要keepdims
        dists = np.add(dists,sq1)
        dists = np.add(dists,sq2)
        return np.sqrt(dists)
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pre = np.zeros(num_test)
        #貌似numpy中没有对多维数组元素出现个数计数的函数，所以只能用循环挨个处理一维数组
        for i in range(num_test):
            pre = self.Ytr[np.argsort(dists[i,:])[:k]]
            pred = np.argmax(np.bincount(pre))         #这两个函数一般配合使用，取出出现次数最多的元素,若有相同会优先取较小的值
            y_pre[i] = pred
        return y_pre