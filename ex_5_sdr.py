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
from lib.sdr import sdr_complete_noise
from lib.classic_mds import classic_mds

############################ PARAMETERS ################################

# number of satellites
n_sats = 10
# satellite noise added to ranges, uniformly sampled between min and max
# noise = (0.,0.)
# noise = (-2.,2.)
noise = (-8.,8.)
# noise = (-50.,50.)

########################################################################

# dimensionality of euclidean space
dims = 3

def rs_mask(num_r,num_s):
    W = np.zeros((num_r+num_s,num_r+num_s))
    W[:num_r,num_r:] = np.ones((num_r,num_s))
    W = W.T + W
    W[num_r:,num_r:] = np.ones((num_s,num_s))
    return W

def graph(R,S,R_sdr):
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
    ax.scatter(R_sdr[0,:],R_sdr[1,:],R_sdr[2,:],label="sdr complete")
    X = np.hstack((R[0,:],R_sdr[0,:]))
    Y = np.hstack((R[1,:],R_sdr[1,:]))
    Z = np.hstack((R[2,:],R_sdr[2,:]))
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
W = rs_mask(R.shape[1],S.shape[1])

# sdr complete
D = sdr_complete_noise(D,W,dims)

X = classic_mds(D,dims)
X = align(X,X[:,R.shape[1]:],S)


graph(R,S,X[:,:R.shape[1]])
