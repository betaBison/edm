#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         25 Jan 2021
# Desc:         alternating descent algorithm
########################################################################

import numpy as np

def alternating_descent(D,W,dims):
    """
    Desc:
        creates mask matrix
    Input(s):
        D (numpy array: n x n) : a noisy, incomplete EDM
        W (numpy array: n x n) : mask matrix
        dims (int) : dimensionality of the euclidean space
    Output(s):
        X (numpy array: d x n) : array of point coordinates
    """

    ########################## PARAMETERS ##############################

    MAX_ITER = 200

    CONVERGE_DISTANCE = 1E-10

    ####################################################################

    n = D.shape[0]              # number of points
    T = 0                       # current epoch
    X = np.zeros((dims,n))      # points initialization
    X_old = np.inf*np.ones((dims,n))

    # centering matrix
    L =  np.eye(n) - (1./n)*np.ones((n,n))


    while T < MAX_ITER:         # loop over iterations
        for i in range(n):      # loop over points
            X_connected = X[:,np.where(W[i,:] == 1)[0]].copy()
            ni = X_connected.shape[1]
            D_connected = D[i,np.where(W[i,:] == 1)[0]].copy()

            for k in range(dims):  # loop over dimensions
                t1 = (-X_connected[k,:]+X[k,i])
                t2 = np.sum((np.tile(X[:,i].reshape(-1,1),(1,ni)) - X_connected)**2,axis=0)
                a = 4.*ni
                b = 12.*np.sum(t1)
                c = 4.*np.sum(t2 + 2.*t1**2 - D_connected)
                d = 4.*np.sum(np.multiply(t1,(t2-D_connected)))

                # only keep real roots
                roots = np.roots([a,b,c,d])
                roots = np.real(roots[np.where(np.abs(np.imag(roots)) < 1E-15)[0]])

                if len(roots) == 0:
                    continue

                cc = np.zeros((1,len(roots)))
                for ii in range(len(roots)):
                    cc[0,ii] = np.sum(((-X_connected[k,:] + X[k,i] + roots[ii])**2 \
                             + t2 - t1**2 - D_connected)**2)

                deltaX = roots[np.argmin(cc)]
                X[k,i] += deltaX

                # center new X
                X = X.dot(L)

        if np.linalg.norm(X - X_old) < CONVERGE_DISTANCE:
            print("converged after",T,"iterations, breaking...")
            break
        else:
            T += 1
            X_old = X.copy()

    if T == MAX_ITER:
        print("forced break after",T,"iterations")

    return X
