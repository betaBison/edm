#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         Classic MDS
########################################################################

import numpy as np

def classic_mds(D,d):
    """
    Desc:
        classic MDS algorithm
    Input(s):
        D (numpy array: n x n) : an EDM where n is the number of points
        d (int) : dimensionality of the euclidean space
    Output(s):
        X (numpy array: d x n) : array of point coordinates
    """
    n = D.shape[0]
    I = np.eye(n)
    J = I - (1./n)*np.ones((n,n))

    G = -0.5*J.dot(D).dot(J)

    U, S, V = np.linalg.svd(G)
    S = np.diag(S)[:d,:]
    X = np.sqrt(S).dot(U.T)
    return X
