#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         08 Feb 2021
# Desc:         compare least squares with EDM methods
#               Problem: compare effect of multipath on solutions
########################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.edm import *
from lib.utils import *
from lib.classic_mds import classic_mds
from lib.least_squares import least_squares
from lib.alternating_rank import rank_complete
from lib.alternating_descent import alternating_descent, alt_ad

############################ PARAMETERS ################################

# number of satellites
n_sats = 9

# MAXIMUM ITERATIONS
max_its = 500

########################################################################

# dimensionality of euclidean space
dims = 3

def graph_ranges(ax,R,S,ranges):

    X = np.hstack((R[0,:],S[0,:]))
    Y = np.hstack((R[1,:],S[1,:]))
    Z = np.hstack((R[2,:],S[2,:]))

    for ss in range(S.shape[1]):
        dir = S[:,ss] - R[:,0]
        end = S[:,ss] - ranges[ss]*dir/np.linalg.norm(dir)
        ax.plot([S[0,ss],end[0]],
                 [S[1,ss],end[1]],
                 [S[2,ss],end[2]])
        ax.scatter(S[0,ss],S[1,ss],S[2,ss],label="satellite" + str(ss))

    ax.scatter(R[0,:],R[1,:],R[2,:],label="receiver truth")

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

def graph_estimates(ax,R,R_ls,R_mds,R_ad):

    # receivers plot
    ax.scatter(R[0,:],R[1,:],R[2,:],label="receivers truth")
    ax.scatter(R_ls[0,:],R_ls[1,:],R_ls[2,:],label="least squares")
    ax.scatter(R_mds[0,:],R_mds[1,:],R_mds[2,:],label="mds")
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

def error(truth,estimate):
    return np.mean(np.linalg.norm(truth-estimate,axis=0)**2)

def sstress(W,X,D_est):
    return np.linalg.norm(np.multiply(W,edm(X) - D_est),'fro')

def circle_points(r,n):
    """
    Desc:
        returns points evenly spaced around a circle
    Input(s):
        r (float) : radius of circle
        n (int) : number of points for circle
    Output(s):
        circle_pts (np array: [2 x n]) : x and y positions around circle
    """
    t = np.linspace(0.,2.*np.pi*(n-1)/(n),n)
    circle_pts = np.array([r*np.cos(t),
                           r*np.sin(t)])
    return circle_pts


########################################################################

# receiver positions
R = np.array([[0.],
              [0.],
              [0.]])

# satellite positions
S = np.zeros((dims,n_sats))
S[:2,:] = circle_points(10,n_sats) \
        + np.random.uniform(low=-2.,high=2.,
                            size=(2,n_sats))
S[2,:] = np.random.uniform(low=11.,high=5.,
                    size=(1,n_sats))

# make ranges and EDM
pranges = np.linalg.norm((S-R), axis = 0)
D = rs_edm_from_ranges(S,pranges)

# receivers plot
fig = plt.figure()
ax1 = fig.add_subplot(221, projection="3d")
graph_ranges(ax1,R,S,pranges)

# least squares
R_ls = least_squares(D,S,max_its)

# mds
X_mds = classic_mds(D,dims)
X_mds = align(X_mds,X_mds[:,1:,],S)
R_mds = X_mds[:,0:1]

# alternating descent
W = np.ones(D.shape)
X_ad,T = alt_ad(D,W,dims,S,max_its)
X_ad = align(X_ad,X_ad[:,1:],S)
R_ad = X_ad[:,0:1]

# estimates plot
ax2 = fig.add_subplot(222, projection="3d")
graph_estimates(ax2,R,R_ls,R_mds,R_ad)


# induce noise
noise = np.random.uniform(low=-1.,high=1.,
                             size=(n_sats))
pranges += noise

# make EDM
D = rs_edm_from_ranges(S,pranges)

# receivers plot
ax3 = fig.add_subplot(223, projection="3d")
graph_ranges(ax3,R,S,pranges)

# least squares
R_ls = least_squares(D,S,max_its)
print("LS error:",error(R,R_ls))

# mds
X_mds = classic_mds(D,dims)
X_mds = align(X_mds,X_mds[:,1:,],S)
R_mds = X_mds[:,0:1]
print("MDS error:",error(R,R_mds))

# alternating descent
W = np.ones(D.shape)
X_ad,T = alt_ad(D,W,dims,S,max_its)
X_ad = align(X_ad,X_ad[:,1:],S)
R_ad = X_ad[:,0:1]
print("AD error:",error(R,R_ad))

# estimates plot
ax4 = fig.add_subplot(224, projection="3d")
graph_estimates(ax4,R,R_ls,R_mds,R_ad)

# give everything enough room
fig.set_size_inches(16, 9)
fig.tight_layout()
# graph all results
plt.show()
