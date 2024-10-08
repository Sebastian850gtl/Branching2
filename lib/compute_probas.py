# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:50:32 2024

@author: sebas
"""
# Uses sizes comming from a Brownian model to compute probabilities

import numpy as np

def state(sample_times,times):
    """
    times : n_times
    sample_times : n_samples x n_clusters

    returns matrix : n_samples x n_times
    """
    times = np.array(times)
    n_times = len(times)
    n_samples,n_clusters = sample_times.shape
    ind_Tt = np.zeros([n_samples,n_times],dtype = 'int')
    for id_sample in range(n_samples):
        sample_T = sample_times[id_sample,:].copy()
        sample_T = np.concatenate((sample_T,np.array([np.inf])))
        i = 0
        s = 0
        for ind_t,t in enumerate(times):
            while s <= t and i < n_clusters:
                i = i + 1
                s = sample_T[i]
            ind_Tt[id_sample,ind_t] = i - 1
    return ind_Tt
            

def number_of_masses_bigger_than_x(arr,x):
    def count(D1array):
        return len(np.where(D1array >= x)[0])
    return np.apply_along_axis(count,arr = arr,axis = 1)
    
    
def probs(sample_sizes,sample_times,times,k,x):
    """ Returns the probability to have more or equal than k cluster of size >= size at
    time t (t can also be an array)"""
    
    ind_Tt = state(sample_times,times)
    n_times = len(times)
    n_samples,_ = sample_times.shape
    #UUt = np.zeros([n_samples,n_times,n])
    sample_indices = np.arange(n_samples)
    P = []
    for i in range(n_times):
        #sample_sizes_t = np.zeros([n_samples,n_clusters])
        sample_sizes_t = sample_sizes[sample_indices,ind_Tt[sample_indices,i],:]
        #print(sample_sizes_t.shape)

        sample_number = number_of_masses_bigger_than_x(sample_sizes_t,x)
        p = len(np.where(sample_number >= k)[0])/n_samples
        P.append(p)
    return np.array(P)

def prob_fun(sample_sizes,sample_times,times,k,Nx):
    """ Returns the probability to have more or equal than k cluster of size >= size at
    time t (t can also be an array)"""

    def count(X,Nx):
        """ Sorted vector X returns vector of dim NX + 2 """
        CS = np.cumsum(X)
        CS[-1] = 1
        X = CS.copy()
        X[1:] = CS[:-1]
        X[0] = 0

        diff = CS - X
        repeats = np.array(diff*(Nx + 1),dtype = 'int32')

        res = np.repeat(np.arrange(X.size),repeats=repeats)
    

    # A reprendre ICI.

    sample_sizes = np.sort(sample_sizes,axis = - 1)
    n_samples,_ = sample_times.shape
    #sample_sizes_min = sample_sizes[:,:,0]
    for i in range(Nx):
        sample_i = np.where(sample_sizes >= i/Nx)
    ind_Tt = state(sample_times,times)
    n_times = len(times)
    n_samples,_ = sample_times.shape
    #UUt = np.zeros([n_samples,n_times,n])
    sample_indices = np.arange(n_samples)
    P = []
    for i in range(n_times):
        #sample_sizes_t = np.zeros([n_samples,n_clusters])
        sample_sizes_t = sample_sizes[sample_indices,ind_Tt[sample_indices,i],:]
        #print(sample_sizes_t.shape)

        sample_number = number_of_masses_bigger_than_x(sample_sizes_t,x)
        p = len(np.where(sample_number >= k)[0])/n_samples
        P.append(p)
    return np.array(P)