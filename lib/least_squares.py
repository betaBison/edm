#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         25 Jan 2021
# Desc:         use least squares to independently localize
#                   receivers and then calculate ranges
########################################################################

import time
import numpy as np

def least_squares(D,S,max_its=20,intermidate_solutions=False):
    """
    Desc:
        creates least squares position solution from EDM
    Input(s):
        D (numpy array: (num_r + num_t) x (num_r + num_t) ) : euclidean
           distance matrix
        S (numpy array: d x num transmitters): locations of transmitters
        max_its (int) : maximum number of iterations
        intermidate_solutions (bool) : return intermidate solutions
    Output(s):
        X_ls (numpy array: d x n) : locations of receivers in graph
    """

    num_s = S.shape[1]
    num_r = D.shape[1] - S.shape[1]
    pranges = np.sqrt(D[:num_r,num_r:])

    X_ls = np.zeros((S.shape[0],num_r))
    X_ls_vec = np.zeros((S.shape[0],num_r,1))
    time_vec = []

    for ii in range(max_its):
        time0 = time.time()
        for rr in range(num_r):
        # x_rr = np.array([0.,0.,0.])
            dist = np.sqrt((X_ls[0,rr]-S[0,:])**2 + (X_ls[1,rr]-S[1,:])**2 + (X_ls[2,rr]-S[2,:])**2)
            pranges_rr = pranges[rr,:]
            G = np.zeros((num_s,3))

            for ss in range(num_s):
                G[ss,0] = (S[0,ss]-X_ls[0,rr])/dist[ss]
                G[ss,1] = (S[1,ss]-X_ls[1,rr])/dist[ss]
                G[ss,2] = (S[2,ss]-X_ls[2,rr])/dist[ss]

            delta = np.linalg.pinv(G).dot(dist - pranges_rr)
            X_ls[:,rr] += delta
        time_vec.append(time.time()-time0)
        X_ls_vec = np.concatenate((X_ls_vec,np.expand_dims(X_ls.copy(),axis=2)),axis=2)

    if intermidate_solutions:
        return X_ls_vec, time_vec
    else:
        return X_ls
