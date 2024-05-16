# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:34:48 2024

@author: sebas
"""

import numpy as np
from time import time
#%% core functions
# See similar functions description in Brownian
def auxl(darr1):
        return np.tril(darr1,k = -1).T
def auxu(darr1):
        return np.triu(darr1,k = 1)

def apply_to_couples(array,fun):
    n = array.size
    triu_indices = np.triu_indices(n,k=1)
    x = np.apply_along_axis(auxu,axis = 0,arr = array)[triu_indices]
    y = np.apply_along_axis(auxl,axis = 0,arr = array)[triu_indices]
    return fun(y,x),triu_indices

#%% class
class Coalescence:
    def __init__(self,n_clusters,kernel):
        self.n_clusters = n_clusters
        self.kernel = np.vectorize(kernel)
        
        self.active_nodes = None
        self.current_sizes = None
        self.current_times = None
        
        self.sample_sizes = None
        self.sample_times = None
        
    def update(self):
        array = self.current_sizes[self.active_nodes]
        kernels,triu_indices = apply_to_couples(array,self.kernel)
        sum_kernels = np.sum(kernels)
        probabilities = kernels/sum_kernels
        
        self.current_times = self.current_times + np.random.exponential(1/sum_kernels)
        
        len_array = triu_indices[0].size
        ind_couple = np.random.choice(len_array,p = probabilities)
        i, j = triu_indices[0][ind_couple], triu_indices[1][ind_couple]
        
        array[i] = array[i] + array[j]
        array[j] = 0
        self.current_sizes[self.active_nodes] = array
        self.active_nodes.pop(j)


    def run(self,n_samples,init = None,save_name = None):
        """  Runs the model
        Arguments:
            n_samples : Number  of sammples
            init : Intialisation of the cluster sizes if set  to False they each receive size 1
            save__name : File name where to save results 
        Results saved as npy file:
            _times : numpy ndarray shape = (n_samples,n_clusters) ; For each sample gives all the collision times, the first column is zeros since there is n_clusters-1 collisions.
            _sizes : numpy ndarray shape = (n_samples,n_clusters,n_clusters); For T =  _times[i,j] the time at sample  i and collision j, _sizes[i,j,:]  is the cluster distribution at sample  i and collision number j.
        """
        self.sample_sizes = np.zeros([n_samples,self.n_clusters,self.n_clusters])
        self.sample_times = np.zeros([n_samples,self.n_clusters])
        
        t0 = time()
        for idi in range(n_samples):
            try: 
                if init.ndim == 1:
                    self.current_sizes = init.copy()
                elif init.ndim == 2:
                     self.current_sizes = init[idi,:].copy()
            except:
                 self.current_sizes = np.ones([self.n_clusters])
            self.active_nodes = list(range(self.n_clusters))
            self.current_times = 0
            
            self.sample_sizes[idi,0,:] = self.current_sizes
            self.sample_times[idi,0] = self.current_times
            for k in range(self.n_clusters-1):
                self.update()
                self.sample_sizes[idi,k+1,:] = self.current_sizes
                self.sample_times[idi,k+1] = self.current_times

            print('\r',   'Advancement : %.1f'%(((idi+1)/n_samples)*100)+' %', 'done in %.2fs.' % (time() - t0),end='')
        print("End")
        print("Saving samples")
        if save_name:
            try:
                previous_save_sizes = np.load(save_name+'_sizes.npy')
                previous_save_times = np.load(save_name+'_times.npy')
                np.save(save_name+'_sizes.npy',np.concatenate((self.sample_sizes,previous_save_sizes),axis = 0))
                np.save(save_name+'_times.npy',np.concatenate((self.sample_times,previous_save_times),axis = 0))
            except:
                np.save(save_name+'_sizes.npy',self.sample_sizes)
                np.save(save_name+'_times.npy',self.sample_times)
        else:
            pass

if __name__ == '__main__':
    # kernel = lambda k,l : 1
    # m = Coalescence(100,kernel)
    # m.run(9)
    
    def kernel(x,y):
        return 'K('+str(x)+','+str(y)+')'
    kernel = np.vectorize(kernel)
    n = 3
    x1 = np.arange(n)
    x = np.apply_along_axis(auxu,axis = 0,arr = x1)[np.triu_indices(n,k=1)]
    y = np.apply_along_axis(auxl,axis = 0,arr = x1)[np.triu_indices(n,k=1)]
    print(kernel(y,x))
    