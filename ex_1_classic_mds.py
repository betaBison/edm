#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         Classic MDS example
########################################################################

import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

from lib.edm import noisy_edm
from lib.classic_mds import classic_mds
from lib.utils import add_all_edges

############################ PARAMETERS ################################

# number of points
n = 5
# standard deviation of noise to add to ranges
# points are sampled between [0,1], so max truth range is 1.0
std_dev = 0.0

########################################################################

# dimensionality of euclidean space
d = 2

def points2graph(X):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i,pos=X[:,i])
    G = add_all_edges(G)
    return G

def compute_mds(X):
    D = noisy_edm(X,std_dev)
    X_mds = classic_mds(D,d)
    return X_mds

def graph(G,ax):
    nx.draw(G, nx.get_node_attributes(G,'pos'),
            font_weight='bold')
    plt.plot([0.,1.],[0.,0.],'r')
    plt.plot([0.,0.],[0.,1.],'g')
    plt.xlim(ax.get_xlim()[0]-0.1,ax.get_xlim()[1]+0.1)
    plt.ylim(ax.get_ylim()[0]-0.1,ax.get_ylim()[1]+0.1)
    ax.set_aspect('equal')

# create plots
fig = plt.figure()
fig.suptitle("Original vs. Classic MDS Reconstruction")

# example 1
X1 = np.vstack((np.arange(n).reshape(1,n)/(n-1),
                np.zeros((1,n))))
X1[0,:] += 0.5*np.random.rand()
X1[1,:] += 0.5*np.random.rand()
graph(points2graph(X1),plt.subplot(321))
graph(points2graph(compute_mds(X1)),plt.subplot(322))

# example 2
theta = 2.*np.pi*np.random.rand()
rot = np.array([[np.cos(theta),-np.sin(theta)],
                [np.sin(theta),np.cos(theta)]])
X2 = rot.dot(X1)
graph(points2graph(X2),plt.subplot(323))
graph(points2graph(compute_mds(X2)),plt.subplot(324))

# example 3
X3 = np.random.rand(d,n)
graph(points2graph(X3),plt.subplot(325))
graph(points2graph(compute_mds(X3)),plt.subplot(326))

fig.tight_layout()
plt.show()
