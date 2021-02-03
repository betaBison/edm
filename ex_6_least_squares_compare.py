#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         25 Jan 2021
# Desc:         compare least squares with EDM methods
#               Problem: localize receivers on the ground
#               Method 1: use least squares to independently localize
#                   receivers and then calculate ranges
#               Method 2: classic mds with full knowledge for comparison
#               Method 3: use alternating_descent to reconstruct ranges
#                   directly
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.utils import *
from lib.edm import edm, rs_edm
from lib.classic_mds import classic_mds
from lib.alternating_descent import alternating_descent

############################ PARAMETERS ################################

# number of satellites
n_sats = 10
# satellite noise added to ranges, uniformly sampled between min and max
# noise = (0.,0.)
noise = (-2.,2.)
# noise = (-8.,8.)
# noise = (-50.,50.)
# number of comparisons to perform (default = 1)
n_compare = 10

########################################################################

# dimensionality of euclidean space
dims = 3

def least_squares(D,S):
    """
    Desc:
        creates least squares position solution from EDM
    Input(s):
        D (numpy array: (num_r + num_t) x (num_r + num_t) ) : euclidean
           distance matrix
        S (numpy array: d x num transmitters): locations of transmitters
    Output(s):
        X_ls (numpy array: d x n) : locations of receivers in graph
    """

    num_s = S.shape[1]
    num_r = D.shape[1] - S.shape[1]
    pranges = np.sqrt(D[:num_r,num_r:])

    X_ls = np.zeros((S.shape[0],num_r))

    for rr in range(num_r):
        x_rr = np.array([0.,0.,0.])
        for ii in range(20):
            dist = np.sqrt((x_rr[0]-S[0,:])**2 + (x_rr[1]-S[1,:])**2 + (x_rr[2]-S[2,:])**2)
            pranges_rr = pranges[rr,:]
            G = np.zeros((num_s,3))

            for ss in range(num_s):
                G[ss,0] = (S[0,ss]-x_rr[0])/dist[ss]
                G[ss,1] = (S[1,ss]-x_rr[1])/dist[ss]
                G[ss,2] = (S[2,ss]-x_rr[2])/dist[ss]

            delta = np.linalg.pinv(G).dot(dist - pranges_rr)
            x_rr += delta
        X_ls[:,rr] = x_rr

    return X_ls

def rs_mask(num_r,num_s):
    W = np.zeros((num_r+num_s,num_r+num_s))
    W[:num_r,num_r:] = np.ones((num_r,num_s))
    W = W.T + W
    W[num_r:,num_r:] = np.ones((num_s,num_s))
    return W

def graph(R,S,R_ls,R_mds,R_ad):
    fig = plt.figure()

    # receivers plot
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(R[0,:],R[1,:],R[2,:],label="receivers truth")
    ax.scatter(S[0,:],S[1,:],S[2,:],label="satellites truth")

    X = np.hstack((R[0,:],S[0,:]))
    Y = np.hstack((R[1,:],S[1,:]))
    Z = np.hstack((R[2,:],S[2,:]))

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.legend()

    # receivers plot
    ax = fig.add_subplot(122, projection="3d")
    ax.scatter(R[0,:],R[1,:],R[2,:],label="receivers truth")
    ax.scatter(R_ls[0,:],R_ls[1,:],R_ls[2,:],label="least squares")
    ax.scatter(R_mds[0,:],R_mds[1,:],R_mds[2,:],label="mds w/ full knowledge")
    ax.scatter(R_ad[0,:],R_ad[1,:],R_ad[2,:],label="alternating descent")
    X = np.hstack((R[0,:],R_ls[0,:],R_mds[0,:],R_ad[0,:]))
    Y = np.hstack((R[1,:],R_ls[1,:],R_mds[1,:],R_ad[1,:]))
    Z = np.hstack((R[2,:],R_ls[2,:],R_mds[2,:],R_ad[2,:]))
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.legend()

    # give everything enough room
    fig.set_size_inches(8, 4.5)
    fig.tight_layout()
    plt.show()

def error(truth,estimate):
    return np.sum(np.linalg.norm(truth-estimate,axis=0)**2)

########################################################################
error_ls = []
error_ad = []
its_ad = []
# for nc in range(n_compare):
while len(error_ad) < n_compare:
    print("Number Compare:",len(error_ad))
    # receiver positions
    R = np.array([[-50.,50.,50.,-50.,0.],
                  [50.,50.,-50.,-50.,0.],
                  [0.,0.,0.,0.,0.]])

    # satellite positions
    S = np.zeros((dims,n_sats))
    S_direction = np.random.rand(dims,n_sats)-0.5
    # force z direction to be positive
    S_direction[2,:] = np.abs(S_direction[2,:])
    # S_distance = np.random.normal(loc=23E6,scale=1E6,size=(n_sats,))
    S_distance = np.random.normal(loc=50,scale=5,size=(n_sats,))

    # normalize to new distance
    for ns in range(n_sats):
        S[:,ns] = S_direction[:,ns] \
                * S_distance[ns]/np.linalg.norm(S_direction[:,ns])

    D = rs_edm(R,S,noise)



    # mds
    truth_edm = edm(np.hstack((R,S)))
    X_mds_full = classic_mds(truth_edm,dims)
    X_mds_full = align(X_mds_full,X_mds_full[:,R.shape[1]:],S)
    R_mds = X_mds_full[:,:R.shape[1]]
    # print("mds error:",error(R,R_mds))

    # alternating descent
    W = rs_mask(R.shape[1],S.shape[1])
    X_ad,t_ad = alternating_descent(D,W,dims)
    X_ad = align(X_ad,X_ad[:,R.shape[1]:],S)
    R_ad = X_ad[:,:R.shape[1]]
    if error(R,R_ad) > 1:
        print("need redo: ",error(R,R_ad))
        continue
    print("alternating descent error:",error(R,R_ad))
    error_ad.append(error(R,R_ad))
    its_ad.append(t_ad)

    # least squares
    R_ls = least_squares(D,S)
    error(R,R_ls)
    print("least squares error:",error(R,R_ls))
    error_ls.append(error(R,R_ls))

    # print("R_ad:",R_ad)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.boxplot([error_ls,error_ad])
plt.xticks([1,2],['least squares','alternating descent'])
plt.title("Summed Error")

ax2 = fig.add_subplot(122)
ax2.boxplot(its_ad)
plt.title("Alternating Descent Iterations")


# graph all results
graph(R,S,R_ls,R_mds,R_ad)
