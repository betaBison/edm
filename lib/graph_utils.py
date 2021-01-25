#!/usr/bin/env python
########################################################################
# Author(s):    D. Knowles
# Date:         24 Jan 2021
# Desc:         Helpful graph utilities
########################################################################

import numpy as np
import networkx as nx

def add_all_edges(G):
    """
    Desc:
        add all edges for nodes in a graph
    Input(s):
        G (networkx graph object) : original graph
    Output(s):
        G (networkx graph object) : final graph
    """
    n = G.number_of_nodes()
    for i in range(n-1):
        for j in range(i+1,n):
            G.add_edge(i,j)
    return G

def update_edges_in_range(G,X,r):
    """
    Desc:
        restrict edges in a graph to within a communication distance
    Input(s):
        G (networkx graph object) : original graph
        X (numpy array: d x n) : locations of nodes in graph
        r (int) : range within edges exist
    Output(s):
        G (networkx graph object) : final graph
    """
    n = X.shape[1]
    for i in range(n-1):
        for j in range(i+1,n):
            if np.linalg.norm(X[:,i] - X[:,j]) <= r:
                G.add_edge(i,j)
            else:
                if (i,j) in G.edges():
                    G.remove_edge(i,j)
    return G

def force_connected(G):
    """
    Desc:
        make sure that all nodes are connected to something
    Input(s):
        G (networkx graph object) : original graph
        n (int) : total number of nodes
    Output(s):
        G (networkx graph object) : final graph
    """
    n = G.number_of_nodes()
    for i in range(n):
        if not nx.has_path(G,source=i,target=0):
            if len(list(G.neighbors(i))) == 0:
                if i != 0:
                    # if it still can't connect to main, connect it!
                    G.add_edge(i,i-1)
                else:
                    G.add_edge(i,n-1)
            if not nx.has_path(G,source=i,target=0):
                # if it still can't connect, force it
                G.add_edge(i,0)
    return G

def mask_matrix(G):
    """
    Desc:
        creates mask matrix
    Input(s):
        G (networkx graph object) : original graph
    Output(s):
        W (numpy array: n x n) : mask matrix
    """
    n = G.number_of_nodes()
    W = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            if (i,j) in G.edges():
                W[i,j] = 1.
                W[j,i] = 1.
    return W
