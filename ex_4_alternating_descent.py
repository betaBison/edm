#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         25 Jan 2021
# Desc:         alternating descent example
########################################################################

import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

from lib.edm import noisy_edm
from lib.utils import *
from lib.alternating_descent import alternating_descent

############################ PARAMETERS ################################

# number of points
n = 7
# standard deviation of noise to add to ranges
# points are sampled between [0,1], so max truth range is 1.0
std_dev = 0.0

########################################################################

# dimensionality of euclidean space
d = 2

def points2graph(X,r):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i,pos=X[:,i])
    G = update_edges_in_range(G,X,r)
    return G

def compute_alt_descent(X,W,d):
    D = np.multiply(W,noisy_edm(X,std_dev))
    X_mds = alternating_descent(D,W,d)
    return X_mds

def graph(G,ax):
    nx.draw(G, nx.get_node_attributes(G,'pos'),
            font_weight='bold')# r = 0.3     # communication range
    plt.plot([0.,1.],[0.,0.],'r')
    plt.plot([0.,0.],[0.,1.],'g')
    plt.xlim(ax.get_xlim()[0]-0.1,ax.get_xlim()[1]+0.1)
    plt.ylim(ax.get_ylim()[0]-0.1,ax.get_ylim()[1]+0.1)
    ax.set_aspect('equal')

# create plots
fig = plt.figure()
fig.suptitle("Original vs. Gradient Descent Reconstruction")

# example 1
r = 1.0    # communication range
X1 = np.random.rand(d,n)
G1 = points2graph(X1,r)
graph(G1,plt.subplot(321))
W1 = mask_matrix(G1)
G1_est = points2graph(compute_alt_descent(X1,W1,d),0.0)
G1_est.add_edges_from(G1.edges())
graph(G1_est,plt.subplot(322))

# example 2
r = 0.8     # communication range
X2 = np.random.rand(d,n)
G2 = points2graph(X2,r)
graph(G2,plt.subplot(323))
W2 = mask_matrix(G2)
G2_est = points2graph(compute_alt_descent(X2,W2,d),0.0)
G2_est.add_edges_from(G2.edges())
graph(G2_est,plt.subplot(324))

# example 3
r = 0.6     # communication range
X3 = np.random.rand(d,n)
G3 = points2graph(X3,r)
graph(G3,plt.subplot(325))
W3 = mask_matrix(G3)
G3_est = points2graph(compute_alt_descent(X3,W3,d),0.0)
G3_est.add_edges_from(G3.edges())
graph(G3_est,plt.subplot(326))

fig.tight_layout()
plt.show()
