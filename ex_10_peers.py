#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         09 Feb 2021
# Desc:         toy peer communication scenario
########################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.edm import *
from lib.utils import *
from lib.sdr import sdr_complete_noise
from lib.classic_mds import classic_mds
from lib.least_squares import least_squares
from lib.alternating_rank import rank_complete
from lib.alternating_descent import alternating_descent, alt_ad

############################ PARAMETERS ################################

# number of satellites
n_sats = 10

# satellite noise added to ranges, uniformly sampled between min and max
# noise_s = (0.,0.)
# noise_s = (-1.,1.)
# noise_s = (-5.,5.)
# noise_s = (-10.,10.)
# noise_s = (-20.,20.)
noise_s = (-10.,10.)

# peer noise added to ranges, uniformly sampled between min and max
# noise_p = (0.,0.)
# noise_p = (-0.1,0.1)
# noise_p = (-0.5,0.5)
# noise_p = (-1.,1.)
noise_p = (-2.,2.)

# MAXIMUM ITERATIONS
max_its = 500

# number of trials
num_trials = 1

# verbose
verbose = True
plot_graphs = True

########################################################################

# dimensionality of euclidean space
dims = 3

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
          '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

def graph_positions(ax,R,S,RS_ranges,
                         P = [], PS_ranges = None, RP_ranges=None):
# def graph_positions(ax,R,S,P = None):

    ax.scatter(R[0,:],R[1,:],R[2,:],label="receiver truth")

    for ss in range(S.shape[1]):
        dir = S[:,ss] - R[:,0]
        end = S[:,ss] - RS_ranges[ss]*dir/np.linalg.norm(dir)
        ax.plot([S[0,ss],end[0]],
                [S[1,ss],end[1]],
                [S[2,ss],end[2]],color=colors[ss+1])
        if len(P) != 0:
            dir = S[:,ss] - P[:,0]
            end = S[:,ss] - PS_ranges[ss]*dir/np.linalg.norm(dir)
            ax.plot([S[0,ss],end[0]],
                    [S[1,ss],end[1]],
                    [S[2,ss],end[2]],color=colors[ss+1])

        ax.scatter(S[0,ss],S[1,ss],S[2,ss],label="satellite" + str(ss),
                   color=colors[ss+1])

    if len(P) != 0:
        ax.scatter(P[0,:],P[1,:],P[2,:],label="peers",color=colors[ss+2])
        dir = P[:,0] - R[:,0]
        end = P[:,0] - RP_ranges[0]*dir/np.linalg.norm(dir)
        ax.plot([P[0,0],end[0]],
                [P[1,0],end[1]],
                [P[2,0],end[2]],color=colors[ss+2])

        X = np.hstack((R[0,:],S[0,:],P[0,:]))
        Y = np.hstack((R[1,:],S[1,:],P[1,:]))
        Z = np.hstack((R[2,:],S[2,:],P[2,:]))
    else:
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
    plt.legend(loc="upper right")

def graph_estimates(ax,R,R_ls,R_mds,R_ad,R_sdr):

    # receivers plot
    ax.scatter(R[0,:],R[1,:],R[2,:],label="receivers truth")
    ax.scatter(R_ls[0,:],R_ls[1,:],R_ls[2,:],label="least squares")
    ax.scatter(R_mds[0,:],R_mds[1,:],R_mds[2,:],label="mds")
    ax.scatter(R_ad[0,:],R_ad[1,:],R_ad[2,:],label="alternating descent")
    ax.scatter(R_sdr[0,:],R_sdr[1,:],R_sdr[2,:],label="sdr")
    X = np.hstack((R[0,:],R_ls[0,:],R_mds[0,:],R_ad[0,:],R_sdr[0,:]))
    Y = np.hstack((R[1,:],R_ls[1,:],R_mds[1,:],R_ad[1,:],R_sdr[1,:]))
    Z = np.hstack((R[2,:],R_ls[2,:],R_mds[2,:],R_ad[2,:],R_sdr[2,:]))

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
              [10.]])

# peer positions
P = np.array([[25.],
              [25.],
              [15.]])

# without peer errors
wop_errors = []
# with peer errors
wp_errors = []

for ttt in range(num_trials):
    if ttt%10 == 0:
        print("ttt: ",ttt)

    # satellite positions
    S = np.zeros((dims,n_sats))
    S_direction = np.random.rand(dims,n_sats)-0.5
    # force z direction to be positive
    S_direction[2,:] = np.abs(S_direction[2,:])
    S_distance = np.random.normal(loc=23E6,scale=1E6,size=(n_sats,))
    # S_distance = np.random.normal(loc=50,scale=20,size=(n_sats,))

    # normalize to new distance
    for ns in range(n_sats):
        S[:,ns] = S_direction[:,ns] \
                * S_distance[ns]/np.linalg.norm(S_direction[:,ns])

    # make ranges and EDM
    RS_ranges = np.linalg.norm((S-R), axis = 0)
    # induce noise
    noise = np.random.uniform(low=noise_s[0],high=noise_s[1],
                                 size=(n_sats))
    RS_ranges += noise
    D = rs_edm_from_ranges(S,RS_ranges)
    # mask matrix
    W = rs_mask(R.shape[1],S.shape[1])

    # receivers plot
    if plot_graphs:
        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection="3d")
        graph_positions(ax1,R,S,RS_ranges)

    if verbose:
        print("\n\nWithout peer knowledge")

    # least squares
    R_ls = least_squares(D,S,max_its)
    if verbose:
        print("LS error:",error(R,R_ls))

    # mds
    X_mds = classic_mds(D,dims)
    X_mds = align(X_mds,X_mds[:,R.shape[1]:,],S)
    R_mds = X_mds[:,0:R.shape[1]]
    if verbose:
        print("MDS error:",error(R,R_mds))

    # alternating descent
    X_ad,T = alt_ad(D,W,dims,S,max_its)
    X_ad = align(X_ad,X_ad[:,R.shape[1]:],S)
    R_ad = X_ad[:,0:R.shape[1]]
    if verbose:
        print("AD error:",error(R,R_ad))

    # sdr complete
    D_sdr = sdr_complete_noise(D,W,dims)
    X_sdr = classic_mds(D_sdr,dims)
    X_sdr = align(X_sdr,X_sdr[:,R.shape[1]:],S)
    R_sdr = X_sdr[:,:R.shape[1]]
    if verbose:
        print("SDR error:",error(R,R_sdr))

    # estimates plot
    if plot_graphs:
        ax2 = fig.add_subplot(222, projection="3d")
        graph_estimates(ax2,R,R_ls,R_mds,R_ad,R_sdr)


    # OPTION WITH PEERS
    if verbose:
        print("\n\nNow with peer knowledge")

    # peer - satellite ranges
    PS_ranges = np.linalg.norm((S-P), axis = 0)
    # induce noise
    noise = np.random.uniform(low=noise_s[0],high=noise_s[1],
                              size=(n_sats))
    PS_ranges += noise

    # peer - receiver ranges
    RP_ranges = np.linalg.norm((R-P), axis = 0)
    # induce noise
    noise = np.random.uniform(low=noise_p[0],high=noise_p[1],
                              size=(P.shape[1]))
    RP_ranges += noise

    # peer EDM
    Dp = rs_edm_from_ranges(S,PS_ranges)

    # receivers plot
    if plot_graphs:
        ax3 = fig.add_subplot(223, projection="3d")
        graph_positions(ax3,R,S,RS_ranges,P,PS_ranges,RP_ranges)

    # least squares
    P_ls = least_squares(Dp,S,max_its)[:,0:1]
    S_ls = np.hstack((P_ls,S))
    S_ls_ranges = np.concatenate((RP_ranges,RS_ranges))
    # peer EDM
    Dr_ls = rs_edm_from_ranges(S_ls,S_ls_ranges)
    Rp_ls = least_squares(Dr_ls,S_ls,max_its)
    if verbose:
        print("LS error:",error(R,Rp_ls))

    # mds
    Dr_mds = Dr_ls.copy()
    # replace with measured ranges
    Dr_mds[R.shape[1],R.shape[1]+P.shape[1]:] = PS_ranges**2
    Dr_mds[R.shape[1]+P.shape[1]:,R.shape[1]] = PS_ranges**2

    Xp_mds = classic_mds(Dr_mds,dims)
    Xp_mds = align(Xp_mds,Xp_mds[:,R.shape[1]+P.shape[1]:],S)
    Rp_mds = Xp_mds[:,0:R.shape[1]]
    if verbose:
        print("MDS error:",error(R,Rp_mds))

    # alternating descent
    W_ad = rs_mask(R.shape[1],S.shape[1]+P.shape[1])
    W_ad[1,1] = 0
    Xp_ad,T = alt_ad(Dr_mds,W_ad,dims,S,max_its)
    Xp_ad = align(Xp_ad,Xp_ad[:,R.shape[1]+P.shape[1]:],S)
    Rp_ad = Xp_ad[:,0:R.shape[1]]
    if verbose:
        print("AD error:",error(R,Rp_ad))

    # sdr complete
    Dp_sdr = sdr_complete_noise(Dr_mds,W_ad,dims)
    Xp_sdr = classic_mds(Dp_sdr,dims)
    Xp_sdr = align(Xp_sdr,Xp_sdr[:,R.shape[1]+P.shape[1]:],S)
    Rp_sdr = Xp_sdr[:,0:R.shape[1]]
    if verbose:
        print("SDR error:",error(R,Rp_sdr))

    # estimates plot
    if plot_graphs:
        ax4 = fig.add_subplot(224, projection="3d")
        graph_estimates(ax4,R,Rp_ls,Rp_mds,R_ad,R_sdr)

        # give everything enough room
        fig.set_size_inches(16, 9)
        fig.tight_layout()


    wop_errors.append([error(R,R_ls),
                       error(R,R_mds),
                       error(R,R_ad),
                       error(R,R_sdr)])
    wp_errors.append([error(R,Rp_ls),
                      error(R,Rp_mds),
                      error(R,Rp_ad),
                      error(R,Rp_sdr)])

wop_errors = np.array(wop_errors)
wp_errors = np.array(wp_errors)

relwop_errors = np.zeros((num_trials,wop_errors.shape[1]-1))
relwp_errors = np.zeros((num_trials,wop_errors.shape[1]-1))

for ti in range(wop_errors.shape[1]-1):
    relwop_errors[:,ti] = np.divide(wop_errors[:,0] - wop_errors[:,ti+1],
                                   wop_errors[:,0])*100.
    relwp_errors[:,ti] = np.divide(wp_errors[:,0] - wp_errors[:,ti+1],
                                   wp_errors[:,0])*100.

print("WOP errors:",np.mean(wop_errors,axis=0))
print("WP errors:",np.mean(wp_errors,axis=0))
print("rel WOP errors:",np.mean(relwop_errors,axis=0))
print("rel WP errors:",np.mean(relwp_errors,axis=0))

box_fig = plt.figure()
box_fig.suptitle(str(num_trials) + " trial(s); " \
                 + str(n_sats) + " satellite(s) with noise ("  \
                 + str(noise_s[0]) + "," + str(noise_s[1]) + "); " \
                 + str(P.shape[1]) + " peer(s) with noise ("  \
                 + str(noise_p[0]) + "," + str(noise_p[1]) + ")")

b1 = box_fig.add_subplot(221)
b1.boxplot(wop_errors)
plt.xticks([1,2,3,4],["LS","MDS","AD","SDR"])
plt.title("Without Peer Errors")

b2 = box_fig.add_subplot(222)
b2.boxplot(relwop_errors)
plt.xticks([1,2,3],["MDS","AD","SDR"])
plt.title("Errors Relative to LS")

b2 = box_fig.add_subplot(223)
b2.boxplot(wp_errors)
plt.xticks([1,2,3,4],["LS","MDS","AD","SDR"])
plt.title("With Peer Errors")

b4 = box_fig.add_subplot(224)
b4.boxplot(relwp_errors)
plt.xticks([1,2,3],["MDS","AD","SDR"])
plt.title("Errors Relative to LS")

# give everything enough room
box_fig.tight_layout()
box_fig.set_size_inches(9, 9)
# plt.subplots_adjust(top=0.9)

# graph all results
plt.show()
