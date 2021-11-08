#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         Classic MDS
########################################################################

import numpy as np

def rank_complete(D,d,max_its=5):
    """
    Desc:
        rank complete algorithm
    Input(s):
        W (numpy array: n x n) : mask matrix
        D (numpy array: n x n) : an EDM where n is the number of points
        d (int) : dimensionality of the euclidean space
        max_its (int) : maximum number of iterations
    Output(s):
        X (numpy array: d x n) : array of point coordinates
    """
    D_orig = D.copy()

    for its in range(max_its):
        U, S, Vh = np.linalg.svd(D)

        S[d+2:] = np.zeros(D.shape[0]-(d+2))
        print("S:",S)
        S = np.diag(S)

        D = U.dot(S).dot(Vh)

        D[1:,1:] = D_orig[1:,1:]
        D[0,0] = 0.0
        print(D)

    print("difference")
    print(D - D_orig)


    n = D.shape[0]
    I = np.eye(n)
    J = I - (1./n)*np.ones((n,n))

    G = -0.5*J.dot(D).dot(J)

    U, S, V = np.linalg.svd(G)
    S = np.diag(S)

    S = S[:d,:]

    X = np.sqrt(S).dot(U.T)
    return X
