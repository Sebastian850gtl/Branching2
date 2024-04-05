# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:34:23 2024

@author: sebas
"""

import numpy as np
from numpy import cos,sin,arcsin,arctan2,arccos,sqrt,pi
from time import time

from scipy.stats import norm
# n_clusters * 3 for dimensionality. 
# In all the code the sphere is centered in 0,0,0 and of radius 1, The affine transformation is done
# only in the class Model.

#%% Core functions,
# All points are on a sphere of center 0,0,0 and radius 1, the domain
# We use physical spherical coordinates azimuth is phi in [0,2*Pi] and elevation is theta in [0,Pi]
# Since the model lies on the upper hemisphere theta is in [0,Pi/2]

def project(array):
    """ Orthogonal projection of all the points on the semi-sphere of radius 1 """
    return np.einsum("ij,i -> ij",array,1/np.sqrt(np.sum(array**2,axis = 1)))

def dotarr(M_arr,v_arr):
    return np.einsum("kij,kj -> ki",M_arr,v_arr)

def cartesian_to_spherical(x,y,z):
    theta = arccos(z)
    phi = arctan2(y,x)
    return theta,phi

def spherical_to_cartesian(theta,phi):
    return sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)

def tangent(array,dB2D):
    """ array = [n_clusters,3]
        subarray = [n_clusters,2]
        for $x$ in array, $y$ in subarray
        $X = [x,y,z]$, 3 dimensional array
        $Y = [y_1,y_2]$, 2 dimensional array
        
        Rotates $[y_1,y_2,0]$ in the plan tagent to the sphere at point $x$"""
    x,y,z = array[:,0],array[:,1],array[:,2]
    theta,phi = cartesian_to_spherical(x,y,z)
    ct,st,cp,sp = cos(theta),sin(theta),cos(phi),sin(phi)

    column1 = np.stack((-sp,cp,np.zeros(phi.shape)),axis = -1)
    column2 = np.stack((-ct*cp,-ct*sp,st),axis = -1)
    rotation_matrix = np.stack((column1,column2),axis = -1)
    return np.einsum('nij,nj -> ni',rotation_matrix,dB2D) + array

def uniform_init(Npoints):
    """ Uniform repartition of polarisome proteins on the tip geometry"""
    Z = np.random.rand(Npoints)
    theta = np.arccos(Z)
    phi = -np.pi + 2*np.pi*np.random.rand(Npoints)
    return np.stack(spherical_to_cartesian(theta,phi),axis = 1)

def boundary(array):
    """ Treats the Boundary condition of points array that went out of the boundary"""
    z = array[:,2]
    ind = np.where(z<0)
    array[ind,2] = - array[ind,2]
    #array_out = array[ind]
    #n = array_out.size
    #if n > 0:
    #    array_out[:,2] = - array_out[:,2]
        
def reflected_brownian_sphere(array,sigmas,dt):
    """ Samples the brownian increment sigmas = [n_samples,n_clusters]"""
    dB = sqrt(dt)*np.einsum("i,ij -> ij",sigmas, np.random.randn(array.shape[0],2))
    U = tangent(array,dB)
    U = project(U)
    boundary(U)
    return U
    
#%% Funcions treating contact
def auxl(darr1):
    return np.tril(darr1,k = -1).T
def auxu(darr1):
    return np.triu(darr1,k = 1)

def apply_to_couples_sum(darr1):
    n = darr1.shape[0]
    darr21 = np.apply_along_axis(auxu,axis = 0,arr = darr1)[np.triu_indices(n,k=1)]
    darr22 = np.apply_along_axis(auxl,axis = 0,arr = darr1)[np.triu_indices(n,k=1)]
    return darr21 + darr22

def apply_to_couples_diff(darr1):
    n = darr1.shape[0]
    darr21 = np.apply_along_axis(auxu,axis = 0,arr = darr1)[np.triu_indices(n,k=1)]
    darr22 = np.apply_along_axis(auxl,axis = 0,arr = darr1)[np.triu_indices(n,k=1)]
    return darr21 - darr22

def search_contact(array,radiuses):
    """ array = [n_samples,n_clusters,3]
        We look for points that are in contact : $|x-y|_2 \leq 2 radius$"""
    
    n_clusters,_ = array.shape
    x, y, z = array[:,0], array[:,1], array[:,2]
    cross_x, cross_y, cross_z = apply_to_couples_diff(x), apply_to_couples_diff(y), apply_to_couples_diff(z)
    dist = cross_x**2 + cross_y**2 + cross_z**2
    try:
        var = radiuses[0]
        cross_radius = apply_to_couples_sum(radiuses)
        ind = np.where((dist < cross_radius**2))
    except:
        ind = np.where((dist < radiuses**2))
    
    return np.triu_indices(n_clusters,k=1)[0][ind],np.triu_indices(n_clusters,k=1)[1][ind]

# Suplementary Functions for Modelv2

def apply_to_couples_sumv2(darr1,triu_indices):
    darr21 = np.apply_along_axis(auxu,axis = 0,arr = darr1)[triu_indices]
    darr22 = np.apply_along_axis(auxl,axis = 0,arr = darr1)[triu_indices]
    return darr21 + darr22

def apply_to_couples_diffv2(darr1,triu_indices):
    darr21 = np.apply_along_axis(auxu,axis = 0,arr = darr1)[triu_indices]
    darr22 = np.apply_along_axis(auxl,axis = 0,arr = darr1)[triu_indices]
    return darr21 - darr22

def compute_cross_radius_cross_sigmas_squares_dist(array,radiuses,sigmas):

    n_clusters,_ = array.shape
    triu_indices = np.triu_indices(n_clusters,k=1)
    x, y, z = array[:,0], array[:,1], array[:,2]
    cross_x, cross_y, cross_z = apply_to_couples_diffv2(x,triu_indices), apply_to_couples_diffv2(y,triu_indices), apply_to_couples_diffv2(z,triu_indices)

    dist = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2) # The distance between clusters 2 by 2

    cross_radiuses = apply_to_couples_sumv2(radiuses,triu_indices) # All the sums of the radiuses 2 by 2

    cross_sigmas_squares = apply_to_couples_sumv2(sigmas**2,triu_indices)
    return dist, cross_radiuses, cross_sigmas_squares,triu_indices
#%%
class Modelv2:
    def __init__(self,n_clusters,sigmafun,radfun):
        self.n_clusters = n_clusters
        self.sigf = np.vectorize(sigmafun)
        self.radiusf = np.vectorize(radfun)
        
        self.dt = 0

        self.times = None
        self.sizes = None
        
        self.active = []
        self.trajectories = None
        
        self.current_position = None
        self.current_sizes = None
        
        self.sample_sizes = None
        self.sample_times = None
        
        return None
    
    def _update_(self,tol):
        # We set the usefull varaibles
        X = self.current_position[self.active,:] # varaible for updating active clusters positionsize = self.current_sizes[self.active]
        size = self.current_sizes[self.active].copy()
        sigmas = self.sigf(size)
        radiuses = self.radiusf(size)
            # Collection of arrays giving for each distinct couples (1,2) : |Z_1 - Z_2|, r1 + r2, sigma1^2 + sigma2^2 
        dist, cross_radiuses, cross_sigmas_squares,triu_indices = compute_cross_radius_cross_sigmas_squares_dist(X,radiuses,sigmas)
        # First step we test if there is any contact at the current step
        
        contact_indices_glob = np.where(dist < cross_radiuses) # The indices in the list of all couple
        #  operation retrieving indices of the colliding couples
        contact_indices_i,contac_indices_j = triu_indices[0][contact_indices_glob], triu_indices[1][contact_indices_glob]
            # If contact.
            # For n > 2 simultaneous collisions we remove n-1 clusters of the active list and the only remaing cluster mass is updated to the sum of
            # the n colliding masses. This treatment is done below: The loop is iterating thgrough an adjacence matrix where adjacent nodes are
            # the indices of colliding clusters.
        if len(contact_indices_i) > 0:
            for i,j in zip(contact_indices_i,contac_indices_j):
                size[j] = size[i] + size[j]
                size[i] = 0
            self.current_sizes[self.active] = size #updates active cluster sizes
            # popping elements in contact_indices_i
            for ki,i in enumerate(contact_indices_i):
                self.active.pop(i-ki) # We have to substract ki because the list self.active looses 1 element at each iteration.

            self.sizes[self.active,:] = np.tile(self.current_sizes,(len(self.active),1))
        else:
            pass

        # Second step we adapt the time step to the new relative poistions
        self._adapt_dt_(tol,cross_sigmas_squares,cross_radiuses,dist)

        # Third and final step, we update the positions of each clusters
        X = self.current_position[self.active,:]
        sigmas = self.sigf(self.current_sizes[self.active])
        U = reflected_brownian_sphere(X,sigmas,self.dt)
        self.current_position[self.active,:] = U
        self.times[self.active] += self.dt #update the clocks of all active clusters 

        return None
        
    
    def _adapt_dt_(self,tol,cross_sigmas_squares,cross_radiuses,dist):
        """ Function adapting the time step to the current realtive cluster positions"""
        alpha = norm.ppf(tol)**(-2)
        #print(alpha)
        dtcircle = np.min(0.01/cross_sigmas_squares)
        dt = np.min(((dist-cross_radiuses*0.9)**2/cross_sigmas_squares))*alpha
        
        self.dt = min(dtcircle,dt)
        return None
        
    def run(self,Ntmax,n_samples = 1,stop = 1,tol = 1e-3,position_init = False,size_init = False
            ,save_trajectories = False, save_name = None):
        
        print("Start of the run number of samples : "+str(n_samples))
        t0 = time()
        self.sample_sizes = np.zeros([n_samples,self.n_clusters,self.n_clusters])
        self.sample_times = np.zeros([n_samples,self.n_clusters])
        for idi in range(n_samples):
            self.active = list(range(self.n_clusters))
            try: 
                size_init.shape
                self.current_sizes = size_init.copy()
            except:
                self.current_sizes = np.ones([self.n_clusters])
            if position_init == 'center':
                Y0 = np.zeros([1,3])
                Y0[0,2] = 1
                self.current_position = np.concatenate((Y0,uniform_init(self.n_clusters-1)),axis = 0)
            elif position_init:
                self.current_position = position_init
            else:
                self.current_position = uniform_init(self.n_clusters)
            if save_trajectories:
                self.trajectories = [self.current_position[self.active]]
            else:
                pass
            
            self.times = np.zeros([self.n_clusters])
            self.sizes = np.tile(self.current_sizes,(self.n_clusters,1))

            k = 0
            while k <= Ntmax and len(self.active) >stop:
                #print(len(self.active))
                k = k + 1
                self._update_(tol)
                if save_trajectories:
                    self.trajectories.append(self.current_position)
                else:
                    pass
            
            sorted_indices = np.argsort(self.times)
            self.times = self.times[sorted_indices]
            self.sizes = self.sizes[sorted_indices,:]
            self.sample_times[idi,:] = self.times
            self.sample_sizes[idi,:,:] = self.sizes
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
# When a cluster becomes inactive its clock stops naturally

class Model:
    def __init__(self,geometry_param,n_clusters,sigma,radius,slow_radius,adapt = True):
        self.n_clusters = n_clusters
        self.g_param = geometry_param
        self.sigma = np.vectorize(sigma)
        self.radius = np.vectorize(radius)
        self.slow_radius = np.vectorize(slow_radius)
        
        self.Rslow = None
        self.dtmin = None
        self.dtmax = None
        self.intdt = None
        
        self.times = None
        self.sizes = None
        
        self.active = None
        self.trajectories = None
        
        self.current_position = None
        self.current_sizes = None
        
        self.sample_sizes = None
        self.sample_times = None
        self.adapt = adapt
        
    def update(self,dt,alpha_min,alpha_max):
        x0,R,phi0 = self.g_param
        X = self.current_position[self.active] # varaible for updating active clusters positionsize = self.current_sizes[self.active]
        sizec = self.current_sizes[self.active]
        radiuses = self.radius(sizec)/R
        contact_indices = search_contact(X,radiuses)
        if contact_indices[0].size > 0:
            self.update_dt_Rslow(alpha_min,alpha_max)
            contact_var = 1
            sizec[contact_indices[0]] = sizec[contact_indices[0]] + sizec[contact_indices[1]]
            sizec[contact_indices[1]] = 0
            self.current_sizes[self.active] = sizec #updates active cluster sizes
            self.active = np.delete(self.active,contact_indices[1]) # removes 1 of the two clusters collinding from active clusters
            self.sizes[self.active,:] = np.tile(self.current_sizes,(len(self.active),1))
            #self.update_dt_Rslow(alpha_min,alpha_max)
        else:
            contact_var = 0
        size = self.current_sizes[self.active]
        sigmas = self.sigma(size)/R
        radiuses = self.radius(size)/R
        X = self.current_position[self.active,:] # varaible for updating active clusters position
        U = reflected_brownian_sphere(X,sigmas,dt)
        self.current_position[self.active,:] = U
        self.times[self.active] += dt #update the clocks of all active clusters 
        return contact_var
        
    def adaptative_update(self,alpha_min,alpha_max):
        X = self.current_position[self.active]
        close_indices = search_contact(X,self.Rslow)  
        slows = np.unique(np.concatenate(close_indices))
        slows_a = self.active[slows]
        self.active = np.delete(self.active,slows) # We update the fast ones

        contact_count = 0
        if len(self.active) > 0:
            contact_count += self.update(self.dtmax,alpha_min,alpha_max)
            fast = self.active.copy()
        else:
            fast = self.active.copy()
        if len(slows)>0:
            self.active = slows_a # We update the slow ones
            i = 0
            while i < self.intdt and self.active.size > 1:
                i = i + 1
                contact_count += self.update(self.dtmin,alpha_min,alpha_max)
            #print(len(close_indices[0]),contact_count)
            #print(self.intdt,i)
            if i<self.intdt:
                contact_count += self.update((self.intdt - i)*self.dtmin,alpha_min,alpha_max)
            self.active = np.concatenate((self.active,fast))
        else:
            pass
        if contact_count > 0 and self.active.size > 1:
            self.update_dt_Rslow(alpha_min,alpha_max)
        else:
            pass
    def update_dt_Rslow(self,alpha_min,alpha_max):
        x0,R,phi0 = self.g_param
        size = self.current_sizes[self.active]
        sigmas = self.sigma(size)/R
        radiuses = self.radius(size)/R
        slow_radiuses = self.slow_radius(radiuses)

        Rslowsum = apply_to_couples_sum(slow_radiuses)
        self.Rslow = np.max(Rslowsum)
        sigsum_square = apply_to_couples_sum(sigmas**2)
        radsum = apply_to_couples_sum(radiuses)
        self.dtmin = np.min((radsum)**2/sigsum_square) * alpha_min
        dtmax = self.Rslow**2 *1/np.max(sigsum_square) * alpha_max
        try :
            self.intdt = int(dtmax/self.dtmin)+1
        except:
            print(size)
            print(sigmas)
        self.dtmax = self.intdt*self.dtmin
        
    def run(self,Ntmax,alpha_min,alpha_max,n_samples = 1,stop = 1,position_init = False,size_init = False
            ,save_trajectories = False, save_name = None):
        
        print("Start of the run number of samples : "+str(n_samples))
        x0,R,phi0 = self.g_param
        t0 = time()
        self.sample_sizes = np.zeros([n_samples,self.n_clusters,self.n_clusters])
        self.sample_times = np.zeros([n_samples,self.n_clusters])
        for idi in range(n_samples):
            self.active = np.arange(self.n_clusters)

            try: 
                size_init.shape
                self.current_sizes = size_init.copy()
            except:
                self.current_sizes = np.ones([self.n_clusters])
            if position_init == 'center':
                Y0 = np.zeros([1,3])
                Y0[0,2] = R
                self.current_position = np.concatenate((Y0,uniform_init(self.n_clusters-1)),axis = 0)
            elif position_init:
                self.current_position = position_init
            else:
                self.current_position = uniform_init(self.n_clusters)
            if save_trajectories:
                self.trajectories = [self.current_position[self.active]]
            else:
                pass
            
            self.times = np.zeros([self.n_clusters])
            self.sizes = np.tile(self.current_sizes,(self.n_clusters,1))

            self.update_dt_Rslow(alpha_min,alpha_max)
            #print('\r',   'Advancement : %.1f'%((idi/n_samples)*100)+' %', 'done in %.2fs.' % (time() - t0),end='')
            k = 0
            if self.adapt:
                while k <= Ntmax and self.active.shape[0] >stop:
                    print(self.active.shape[0])
                    k = k + 1
                    self.adaptative_update(alpha_min,alpha_max)
                    if save_trajectories:
                        self.trajectories.append(self.current_position)
                    else:
                        pass
            else:
                while k <= Ntmax and self.active.shape[0] >stop:
                    k = k + 1
                    self.update(self.dtmin,alpha_min,alpha_max)
                    if save_trajectories:
                        self.trajectories.append(self.current_position)
                    else:
                        pass
            sorted_indices = np.argsort(self.times)
            self.times = self.times[sorted_indices]
            self.sizes = self.sizes[sorted_indices,:]
            self.sample_times[idi,:] = self.times
            self.sample_sizes[idi,:,:] = self.sizes
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
# When a cluster becomes inactive its clock stops naturally