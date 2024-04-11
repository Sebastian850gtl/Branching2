# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:34:23 2024

@author: sebas
"""

import numpy as np
from numpy import cos,sin,arcsin,arctan2,arccos,sqrt,pi
from time import time

from scipy.stats import norm

# The exact increments of Brownian on the sphere
def sample_infinite_descent(t,tol = 1e-1):
    n = int(1/tol*np.max(1/t))
    size = t.shape[0]
    res_array = np.ones([size])*n
    k = n
    n_active = size
    s = t.copy()
    array = np.zeros([size])
    while n_active > 0 and k > 0:
        new_array = array + np.random.exponential(2/(k*(k+1)),n_active)
        
        indices = np.where(new_array < s) # le nombre qui est plus petit que t

        n_active = len(indices[0])
        res_array[indices] = res_array[indices] - 1
        array = new_array[indices]
        s = s[indices]
        k = k - 1
    return res_array

def sample_WF(t,a,b,tol = 1e-1):
    M = sample_infinite_descent(t,tol = tol)
    Y = np.random.beta(a,b+M)
    return Y.reshape(-1)


def sample_exactBr(start,dt,tol = 1e-1):
    size,_ = start.shape
    X = sample_WF(dt,1,1,tol = tol)

    phi = np.random.rand(size)*2*np.pi

    I = np.tile(np.eye(3),(size,1,1))
    ed = np.concatenate((np.zeros([size,2]),np.ones([size,1])),axis = 1)
    u = project(ed-start)
    O = I - 2*np.einsum('ij,ik-> ijk',u,u)

    var = 2 * np.sqrt(X*(1-X))
    first_coordinate,second_coordinate,third_coordinate  = var *cos(phi) , var * sin(phi), 1 - 2*X
    vec = np.stack((first_coordinate,second_coordinate,third_coordinate),axis = 1)

    res = np.einsum('ijk,ik-> ij',O,vec)
    #print(np.sum(res**2)/size)
    return res
#%% Core functions,
# All points are on a sphere of center 0,0,0 and radius 1
# We use physical spherical coordinates azimuth is phi in [0,2*Pi] and elevation is theta in [0,Pi]
# The clusters coordinates lie on the upper hemisphere therefore theta is in [0,Pi/2]

def project(array):
    """ Orthogonal projection of all the points on the semi-sphere of radius 1 """
    return np.einsum("ij,i -> ij",array,1/np.sqrt(np.sum(array**2,axis = 1)))


def cartesian_to_spherical(x,y,z):
    theta = arccos(z)
    if any(np.isnan(theta)):
        print(z)
        print(theta)
        print(type(z))
    phi = arctan2(y,x) # equal to arctan(y/x) when x is not 0 and y>0, arctan(y/x) + pi when x not 0 and y <= 0. 
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
    # A = rotation matrix, ker(A) is of dimension 1 and generated by the unit norm orthogonal vector to the
    # tangent plane
    # returns A dB + array , affine transformation of the 2D Brownian increment which is equivalent to
    # a 2D Brownian increment in the Affine plan of parameters (ker(A),array)
    return np.einsum('nij,nj -> ni',rotation_matrix,dB2D) + array

def uniform_init(Npoints,radius0):
    """ Uniform repartition of polarisome proteins on the tip geometry"""
    #Z = np.random.rand(Npoints)*(1-radius0) + radius0
    Z = np.random.rand(Npoints)
    theta = np.arccos(Z)
    phi = -np.pi + 2*np.pi*np.random.rand(Npoints)
    return np.stack(spherical_to_cartesian(theta,phi),axis = 1)

def boundary(array,radiuses):
    """ Treats the Boundary condition of points array that went out of the boundary"""
    z = array[:,2]# - radiuses
    ind = np.where(z<0)
    array[ind,2] =  - array[ind,2] #+ 2*radiuses[ind]
    #array_out = array[ind]
    #n = array_out.size
    #if n > 0:
    #    array_out[:,2] = - array_out[:,2]

def reflected_brownian_sphere_old(array,sigmas,dt,radiuses):
    dB = sqrt(dt)*np.einsum("i,ij -> ij",sigmas, np.random.randn(array.shape[0],2))
    U = tangent(array,dB)
    U = project(U)
    boundary(U,radiuses)
    return U
        
def reflected_brownian_sphere(array,sigmas,dt,radiuses,switch = 0.05):
    """ Samples the brownian increment sigmas = [n_samples,n_clusters]"""
    n_clusters,_ = array.shape
    dtsigmas = np.sqrt(dt)*sigmas

    itangent = np.where(dtsigmas < switch)
    iexact = np.where(dtsigmas >= switch)
    res = np.zeros([n_clusters,3])

    array_tangent = array[itangent]
    array_exact = array[iexact]

    dB = sqrt(dt)*np.einsum("i,ij -> ij",sigmas[itangent], np.random.randn(array_tangent.shape[0],2))
    U = tangent(array_tangent,dB)
    U = project(U)
    res[itangent] = U
    if len(iexact[0])>0:
        res[iexact] = sample_exactBr(array_exact,dt*sigmas[iexact],tol = 1)
    boundary(res,radiuses)
    return res
    
#%% Funcions treating contact
def auxl(darr1):
    return np.tril(darr1,k = -1).T
def auxu(darr1):
    return np.triu(darr1,k = 1)

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
    """ Simulation of the Browninan Coalescence on the surface of a semi-sphere of radius 1 with reflective boundary conditions."""
    def __init__(self,n_clusters,sigmafun,radfun):
        """
        n_clusters, integer : Number of initial clusters.
        sigmafun : Standard deviation function : $\sigma(x) = \sqrt{2D(x)}$ where $D$ is the diffusion function.
        radfun : Radius function.
        """
        self.n_clusters = n_clusters
        self.sigf = np.vectorize(sigmafun)
        self.radiusf = np.vectorize(radfun)
        
        self.dt = 0 # Adaptative time step

        self.times = None # shape = n_clusters
        self.sizes = None # shape = n_clusters x n_clusters 
        
        self.active = [] # List of active clusters. It is initialized in the run method not here.
        self.trajectories = None # Tab of trajectories, only usfull for visual reprenstations.
        
        self.current_position = None # The current positions in cartesian coordiantes of the clusters.
        self.current_sizes = None # The current sizes of all clusters, the size is a positive real value.
        
        self.sample_sizes = None # 3D array n_samples x n_clusters x n_clusters, that the run method will save.
        self.sample_times = None # 2D array n_samples x n_clusters, second return of the run method.
        
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
        self._adapt_dt_(tol,cross_sigmas_squares = cross_sigmas_squares,cross_radiuses = cross_radiuses,dist = dist)

        # Third and final step, we update the positions of each clusters
        X = self.current_position[self.active,:]
        newsizes = self.current_sizes[self.active] 
        sigmas = self.sigf(newsizes)
        radiuses = self.radiusf(newsizes)
        U = reflected_brownian_sphere(X,sigmas,self.dt,radiuses)
        self.current_position[self.active,:] = U
        self.times[self.active] += self.dt #update the clocks of all active clusters 

        return None
        
    
    def _adapt_dt_(self,tol,cross_sigmas_squares,dist,cross_radiuses):
        """ Function adapting the time step to the current realtive cluster positions"""
        alpha = norm.ppf(tol)**(-2)
        #print(alpha)
        dtcircle = 2#np.min(0.004/cross_sigmas_squares)
        dt = np.min(((dist)**2/cross_sigmas_squares))*alpha
        
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

            radius0 = np.min(self.radiusf(self.current_sizes))
            if position_init == 'center':
                Y0 = np.zeros([1,3])
                Y0[0,2] = 1
                self.current_position = np.concatenate((Y0,uniform_init(self.n_clusters-1,0)),axis = 0)
            elif position_init:
                self.current_position = position_init
            else:
                self.current_position = uniform_init(self.n_clusters,radius0)
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