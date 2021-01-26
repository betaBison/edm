#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         creates EDM from points
########################################################################

import numpy as np

def edm(X):
    """
    Desc:
        creates edm from points
    Input(s):
        X (numpy array: d x n) : locations of nodes in graph
    Output(s):
        D (numpy array: n x n) : euclidean distance matrix
    """
    n = X.shape[1]
    G = (X.T).dot(X)
    D = np.diag(G).reshape(-1,1).dot(np.ones((1,n))) \
        - 2.*G + np.ones((n,1)).dot(np.diag(G).reshape(1,-1))
    return D

def noisy_edm(X,std_dev):
    """
    Desc:
        creates edm from points
    Input(s):
        X (numpy array: d x n) : locations of nodes in graph
        std_dev (float) : standard deviation of noise for ranges
    Output(s):
        D (numpy array: n x n) : euclidean distance matrix
    """
    n = X.shape[1]
    D = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            sqrd_dist = (np.linalg.norm(X[:,i]-X[:,j]) \
                         + np.random.normal(scale=std_dev))**2
            D[i,j] = sqrd_dist
            D[j,i] = sqrd_dist
    return D

def rs_edm(R,S,noise):
    """
    Desc:
        creates edm from receiver and transmitter points
    Input(s):
        R (numpy array: d x num receivers) : locations of receivers
        S (numpy array: d x num transmitters): locations of transmitters
        noise (tuple (min,max)) : noise ranges, uniform sample
    Output(s):
        D (numpy array: (num_r + num_s) x (num_r + num_s) ) : euclidean
            distance matrix
    """
    num_r = R.shape[1]
    num_s = S.shape[1]

    RS_pre = np.diag(R.T.dot(R)).reshape(-1,1).dot(np.ones((1,num_s))) \
       - 2.*R.T.dot(S) \
       + np.ones((num_r,1)).dot(np.diag(S.T.dot(S)).reshape(1,-1))
    RS_noise = np.random.uniform(low=noise[0],high=noise[1],
                            size=(num_r,num_s))**2
    RS = np.add(RS_pre,RS_noise)
    D = np.zeros((num_r+num_s,num_r+num_s))
    D[:num_r,num_r:] = RS
    D = D.T + D

    return D
