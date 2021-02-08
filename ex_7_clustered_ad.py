#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         03 Feb 2021
# Desc:         compare least squares with EDM methods
#               Problem: localize receivers on the ground
#               Method 1: use least squares to independently localize
#                   receivers and then calculate ranges
#               Method 2: classic mds with full knowledge for comparison
#               Method 3: use alternating_descent to reconstruct ranges
#                   directly
########################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.utils import *
from lib.edm import edm, rs_edm
from lib.classic_mds import classic_mds
from lib.least_squares import least_squares
from lib.alternating_descent import alternating_descent, alt_ad

############################ PARAMETERS ################################

# number of satellites
n_sats = 10
# satellite noise added to ranges, uniformly sampled between min and max
# noise = (0.,0.)
# noise = (-2.,2.)
noise = (-8.,8.)
# noise = (-50.,50.)
# number of comparisons to perform (default = 1)
n_compare = 1
# MAXIMUM ITERATIONS
max_its = 50

########################################################################

# dimensionality of euclidean space
dims = 3

def rs_mask(num_r,num_s):
    W = np.zeros((num_r+num_s,num_r+num_s))
    W[:num_r,num_r:] = np.ones((num_r,num_s))
    W = W.T + W
    W[num_r:,num_r:] = np.ones((num_s,num_s))
    return W

def error(truth,estimate):
    return np.mean(np.linalg.norm(truth-estimate,axis=0)**2)

def sstress(W,X,D_est):
    return np.linalg.norm(np.multiply(W,edm(X) - D_est),'fro')

########################################################################
error_ls = []
error_ad = []
its_ad = []
for nc in range(n_compare):
# while len(error_ad) < n_compare:
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

    # alternating descent
    W = rs_mask(R.shape[1],S.shape[1])
    X_ad_vec,t_ad,ad_time_vec = alt_ad(D,W,dims,S,max_its,True)
    R_ad_errors = []
    sstress_ad = []
    for its in range(X_ad_vec.shape[2]):
        X_ad = align(X_ad_vec[:,:,its],X_ad_vec[:,R.shape[1]:,its],S)
        R_ad = X_ad[:,:R.shape[1]]
        R_ad_errors.append(error(R,R_ad))
        sstress_ad.append(sstress(W,X_ad,D))
    # if error(R,R_ad) > 1:
        # print("need redo: ",error(R,R_ad))
        # continue
    print("alternating descent error:",R_ad_errors[-1])
    # its_ad.append(t_ad)

    # clustered alternating_descent
    num_r = R.shape[1]
    R_alt_ad = np.zeros((dims,num_r,max_its+1))
    sstress_alt_ad = []
    time0 = time.time()
    for rad_i in range(num_r):
        Dr_1 = np.hstack((np.array([[D[rad_i,rad_i]]]),
                          D[rad_i,num_r:].reshape(1,-1)))
        Dr_2 = np.hstack((D[num_r:,rad_i].reshape(-1,1),
                          D[num_r:,num_r:]))
        Dr = np.vstack((Dr_1,Dr_2))
        Wr = rs_mask(1,S.shape[1])
        X_alt_r,t_ad,tv = alt_ad(Dr,Wr,dims,S,max_its,True)

        for its in range(max_its+1):
            X_alt_r_aligned = align(X_alt_r[:,:,its],X_alt_r[:,1:,its],S)
            R_alt_ad[:,rad_i,its] = X_alt_r_aligned[:,0]
        if rad_i == 0:
            aad_time_vec = [0.0,time.time()-time0]
    R_alt_ad_errors = []
    for its in range(max_its+1):
        R_alt_ad_errors.append(error(R,R_alt_ad[:,:,its]))
        X_alt_ad = np.hstack((R_alt_ad[:,:,its],S))
        sstress_alt_ad.append(sstress(W,X_alt_ad,D))
    print("clustered ad error:",R_alt_ad_errors[-1])

        # print("Dr:",Dr)

    # mds


    # truth_edm = edm(np.hstack((R,S)))
    R_mds = np.zeros(R.shape)
    time0 = time.time()
    for rad_i in range(num_r):
        Dr_1 = np.hstack((np.array([[D[rad_i,rad_i]]]),
                          D[rad_i,num_r:].reshape(1,-1)))
        Dr_2 = np.hstack((D[num_r:,rad_i].reshape(-1,1),
                          D[num_r:,num_r:]))
        Dr = np.vstack((Dr_1,Dr_2))
        X_mds_r = classic_mds(Dr,dims)
        X_mds_r = align(X_mds_r,X_mds_r[:,1:,],S)
        R_mds[:,rad_i] = X_mds_r[:,0]
    # X_mds_full = align(X_mds_full,X_mds_full[:,R.shape[1]:],S)
    # R_mds = X_mds_full[:,:R.shape[1]]
    mds_time = time.time() - time0
    mds_error = error(R,R_mds)
    X_mds_full = np.hstack((R_mds,S))
    sstress_mds = sstress(W,X_mds_full,D)
    print("mds error:",mds_error)


    # least squares
    R_ls_vec,ls_time_vec = least_squares(D,S,max_its,True)
    R_ls_errors = []
    sstress_ls = []
    for its in range(R_ls_vec.shape[2]):
        R_ls_errors.append(error(R,R_ls_vec[:,:,its]))
        X_ls = np.hstack((R_ls_vec[:,:,its],S))
        sstress_ls.append(sstress(W,X_ls,D))
    print("least squares error:",R_ls_errors[-1])
    # error_ls.append(error(R,R_ls))

    # print("R_ad:",R_ad)


fig = plt.figure()
ax1 = fig.add_subplot(221)
plt.plot(range(R_ls_vec.shape[2]),R_ls_errors,label="least squares")
plt.plot(range(X_ad_vec.shape[2]),R_ad_errors,label="alternating descent")
plt.plot(range(max_its+1),R_alt_ad_errors,label="clustered alternating descent")
plt.plot([0,max_its+1],[mds_error,mds_error],label="classical mds")
plt.xlabel("iterations")
plt.title("Mean Squared Error")
plt.legend()

ax2 = fig.add_subplot(222)
plt.plot(range(R_ls_vec.shape[2]),sstress_ls,label="least squares")
plt.plot(range(X_ad_vec.shape[2]),sstress_ad,label="alternating descent")
plt.plot(range(max_its+1),sstress_alt_ad,label="clustered alternating descent")
plt.plot([0,max_its+1],[sstress_mds,sstress_mds],label="classical mds")
plt.title("S-stress")
plt.xlabel("iterations")
plt.legend()

ax3 = fig.add_subplot(223)
plt.plot(range(len(ls_time_vec)),np.cumsum(ls_time_vec),label="least squares")
plt.plot(range(len(ad_time_vec)),np.cumsum(ad_time_vec),label="alternating descent")
plt.plot([0.,max_its-1],np.cumsum(aad_time_vec),label="alt ad")
plt.plot([0.,max_its-1],[mds_time,mds_time],label="classical mds")
plt.legend()

ax4 = fig.add_subplot(224)
plt.plot([0,max_its],[0,0])
ad_data = np.divide(np.array(R_ls_errors) - np.array(R_ad_errors),np.array(R_ls_errors))
plt.plot(range(X_ad_vec.shape[2]),ad_data,label="alternating descent")
alt_ad_data = np.divide(np.array(R_ls_errors) - np.array(R_alt_ad_errors),np.array(R_ls_errors))
plt.plot(range(max_its+1),alt_ad_data,label="clustered alternating descent")
mds_data = np.divide(np.array(R_ls_errors) - mds_error,np.array(R_ls_errors))
plt.plot(range(max_its+1),mds_data,label="classical mds")
plt.ylim([-2,2])
plt.legend()



plt.show()
