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

    For each sample maps the time range onto the sample of jumping times 
    At time t gives sup(i, T_i < t)
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
            while s <= t:# and i < n_clusters:
                i = i + 1
                s = sample_T[i]
            ind_Tt[id_sample,ind_t] = i - 1
    return ind_Tt

def number_of_masses(sample_mass,ind_Tt,masses):
    """
    sorted_sample_mass : shape n_sample x n_clusters x n_clusters, sorted on the last axis
    masses : shape n_masses

    returns array : n_samples x n_times x n_masses
    """
    _,_,n_clusters = sample_mass.shape
    n_samples,n_times = ind_Tt.shape
    n_masses = len(masses)
    numbers = np.zeros([n_samples,n_times,n_masses],dtype = 'int32')
    for id_sample in range(n_samples):
        for ind_t in range(n_times):
            sample_X = np.sort(sample_mass[id_sample,ind_Tt[id_sample,ind_t],:])
            sample_X = np.concatenate((sample_X,np.array([np.inf])))

            s, i = sample_X[0], 0
            for ind_x, x in enumerate(masses):
                while s < x:# and i < n_clusters:
                    i = i + 1
                    s = sample_X[i]
                numbers[id_sample,ind_t,ind_x] = i
    return n_clusters - numbers


def prob_fun(sample_mass,sample_times,times,masses,k):
    """ Returns the probability to have more or equal than k cluster of size >= size at
    time t (t can also be an array)"""
    ind_Tt = state(sample_times,times)
    numbers = number_of_masses(sample_mass,ind_Tt,masses)
    n_samples,n_times,n_masses = numbers.shape
    res = np.zeros([n_masses,n_times])
    for ind_t in range(n_times):
        for ind_x in range(n_masses):
            where_numbers = np.where(numbers[:,ind_t,ind_x] >= k)[0]
            res[ind_x,ind_t] = len(where_numbers)/n_samples
    return res

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

def prob_fun1(sample_sizes,sample_times,times,k,Nx):
    """ Returns the probability to have more or equal than k cluster of size >= size at
    time t (t can also be an array)"""
    n_samples,_ = sample_times.shape
    ind_Tt = state(sample_times,times)
    n_times = len(times)

    #UUt = np.zeros([n_samples,n_times,n])
    sample_indices = np.arange(n_samples)
    
    res = np.zeros([Nx+1,n_times])
    for i in range(n_times):
        print(i)
        #sample_sizes_t = np.zeros([n_samples,n_clusters])
        sample_sizes_t = sample_sizes[sample_indices,ind_Tt[sample_indices,i],:]
        #print(sample_sizes_t.shape)
        for j in range(Nx+1):
            print(i,j)
            x = j/Nx
            sample_number = number_of_masses_bigger_than_x(sample_sizes_t,x)
            p = len(np.where(sample_number >= k)[0])/n_samples
            res[j,i] = p
    return res




