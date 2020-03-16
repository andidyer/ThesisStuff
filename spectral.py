import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json

#Load the embeddings
def load_embeddings(filepath, N=1000):
    """Loads embeddings from given file.  Expects file in word2vec format (with vocab x hidden as header)"""
    with open(filepath) as fin:
        nrow, ndim = fin.readline().strip().split()
        nrow, ndim = int(nrow), int(ndim)
        N = min(N,nrow)
        A = np.zeros((N,ndim))
        for i in range(N):
            line = fin.readline().strip().split()
            word, vec = line[0], line[1:]
            vec = list(map(float, vec))
            A[i] = vec
            words += [word]
    return words, A

def normalise(embs):
    """Normalisation pipeline for embeddings.  Length norm, mean center, length norm"""
    embs /= np.linalg.norm(embs, axis=1)[:,None]
    embs -= np.mean(embs, axis=0)
    embs /= np.linalg.norm(embs, axis=1)[:,None]
    return embs

def similarity(embs, for_Laplacian=True):
    """Gets a self-similarity matrix from a matrix of embeddings"""
    u,s,vt = np.linalg.svd(embs, full_matrices=False)
    sim = np.dot(u*s,u.T)
    #sim = np.dot(embs,embs.T)
    del u, s, vt
    if for_Laplacian:
        for i,_ in enumerate(sim):
            sim[i,i] = -1  #Self should not be most similar
    return sim

def adjacency(sim):
    """Gets adjacency matrix from self-similarity matrix."""
    K = 1
    topK = np.argsort(sim)[:,::-1][:,:K]
    adj = np.zeros(sim.shape)
    for i, inds in enumerate(topK):
        adj[i,tuple(inds)] = 1
    return adj

def degree(adj):
    """Gets degree matrix from adjacency matrix."""
    degree = np.diag(np.sum(adj, axis=0))
    return degree

def laplace(degree, adj):
    """Gets Laplacian matrix from degree matrix and adjacency matrix.  L = D - A"""
    return degree - adj

def l_eigs(laplace):
    """Gets eigenvalues of Laplacian matrix.
        These represent the spectrum of the embedding matrix.
        Sorts values from largest to smallest."""
    eigs, _ = np.linalg.eig(laplace)
    eigs = np.where(eigs>0,eigs,0)
    eigs.sort()
    return eigs[::-1]

def l_sigma(laplace):
    """Gets singular values of Laplacian matrix.
        These represent the spectrum of the embedding matrix.."""
    _,s,_ = np.linalg.svd(laplace)
    return s

def largest_eigs(E, n=0.9):
    assert 0.0 <= n <=1.0, "n must be a real number in the range [0.1]."
    """Gets largest eigen- or singular values corresponding to n of the value mass"""
    outlist = []
    SUM = np.sum(E)
    for i,e in enumerate(E):
        if np.sum(E[:i+1]) / SUM < k:
            outlist.append(e)
    return np.array(outlist)

def get_eigs_from_file(filepath, n=0.9):
    """Pipeline for directly getting largest eigenvalues corresponding to n proportion of eigenvalue mass"""
    words, embs = load_embeddings(filepath, N=1000)
    embs = normalise(embs)
    sim = similarity(embs)
    adj = adjacency(sim)
    deg = degree(adj)
    L = laplace(deg,adj)
    lam = l_eigs(L, n)
    lam = largest_eigs(lam)
    return lam

def compare_L(src_eigs, tgt_eigs):
    """Gets the difference between two sets of Laplacian eigenvalues.
        Where the number of values is different, the two are zipped and the smaller number is dominant.
        Criterion is squared difference.  Might add some others later"""
    return (src_eigs - tgt_eigs)**2

#TODO: Argparse, function for finding difference between many languages

