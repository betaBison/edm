#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         25 Jan 2021
# Desc:         alternating descent algorithm
########################################################################

import time
import cvxpy as cp
import numpy as np

def sdr_complete_noise(D,W,dims):
    """
    Desc:
        creates mask matrix
    Input(s):
        D (numpy array: n x n) : a noisy, incomplete EDM
        W (numpy array: n x n) : mask matrix
        dims (int) : dimensionality of the euclidean space
    Output(s):
        D (numpy array: d x n) : complete EDM
    """

    ########################## PARAMETERS ##############################

    lamb = np.sqrt(np.sum(np.where(W == 0 ,1,0)))

    ####################################################################

    n = D.shape[0]

    x = -1./(n + np.sqrt(n))
    y = -1./np.sqrt(n)
    V = np.vstack((y*np.ones((1,n-1)),
                   x*np.ones((n-1,n-1)) + np.eye(n-1)))
    e = np.ones((n, 1));

    G = cp.Variable((n-1,n-1), PSD=True)
    B = V @ G @ V.T
    E = cp.reshape(cp.diag(B),(n,1)) @ e.T - 2.*B + e @ cp.reshape(cp.diag(B),(1,n))

    objective = cp.Maximize(cp.trace(G) - lamb * cp.norm(cp.multiply(W, (E - D)),"fro"))

    prob = cp.Problem(objective)
    prob.solve()

    G = G.value
    B = V @ G @ V.T
    D = np.diag(B).reshape(-1,1) @ e.T - 2.*B + e @ np.diag(B).reshape(1,-1)

    return D
