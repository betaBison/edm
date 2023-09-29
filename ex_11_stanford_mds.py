#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         Classic MDS example
########################################################################

import os
# import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore", category=UserWarning)

from lib.classic_mds import classic_mds
from lib.utils import add_all_edges

############################ PARAMETERS ################################

# standard deviation of noise to add to ranges
# points are sampled between [0,1], so max truth range is 1.0
std_dev = 0.0

########################################################################

# dimensionality of euclidean space
d = 2

def points2graph(X):
    G = nx.Graph()
    for i in range(X.shape[1]):
        G.add_node(i,pos=X[:,i])
    G = add_all_edges(G)
    return G

labels={}
labels[0]="stanford oval"
labels[1]="green library"
labels[2]="Durand"
labels[3]="huang center"
labels[4]="Ray's Grill"
labels[5]="Hoover Tower"
labels[6]="Memorial Church"
labels_orig = labels.copy()

D = np.loadtxt(os.path.join(os.getcwd(),"data","walking.csv"),
               delimiter=",",dtype=int)
D = np.square(D)
D_orig = D

def graph(G,ax,labels=None):
    nx.draw(G, nx.get_node_attributes(G,'pos'),
            font_weight='bold')
    if labels is not None:
        nx.draw_networkx_labels(G, nx.get_node_attributes(G,'pos'),
                                labels)
    plt.plot([0.,1.],[0.,0.],'r')
    plt.plot([0.,0.],[0.,1.],'g')
    plt.xlim(ax.get_xlim()[0]-0.1,ax.get_xlim()[1]+0.1)
    plt.ylim(ax.get_ylim()[0]-0.1,ax.get_ylim()[1]+0.1)
    ax.set_aspect('equal')

# create plots
fig = plt.figure()
fig.suptitle("Original vs. Classic MDS Reconstruction")

# example 1
X_result = classic_mds(D,d)
graph(points2graph(X_result),plt.subplot(111),labels)

# plot two with labels
fig = plt.figure()
fig.suptitle("Original vs. Classic MDS Reconstruction")

# example with labels
X_result = classic_mds(D,d)
graph(points2graph(X_result),plt.subplot(111))

fig.tight_layout()
plt.show()
