import numpy as np
from numpy.random import beta

def sample_infinite_descent(t,size,tol = 1e-1):
    n = int(tol*t)
    res_array = np.ones([size]*n)
    k = n
    n_active = size
    array = np.zeros([size])
    while n_active > 0 and k >= 0:
        new_array = array + np.exponential(k*(k+1)/2,n_active)

        indices = np.where(array < t) # le nombre qui est plus petit que t

        n_active = len(indices[0])
        res_array[indices] = res_array[indices] - 1
        array = new_array[indices]
        k = k - 1
    return res_array

def sample_WF(t,a,b,size,tol = 1e-1):
    M = sample_infinite_descent(t,size = size,tol = tol)
    Y = beta(a,b+M)
    return Y.reshape(-1)

def project(array):
    """ Orthogonal projection of all the points on the semi-sphere of radius 1 """
    return np.einsum("ij,i -> ij",array,1/np.sqrt(np.sum(array**2,axis = 1)))

def sample_exactBr(start,dt,tol = 1e-1):
    size,_ = start.shape
    X = sample_WF(dt,1,1,size,tol = tol)
    phi = np.random.rand(size)*2*np.pi
    Y = np.stack((np.cos(phi),np.sin(phi)))

    I = np.tile(np.eye(3),(size,3,3))
    ed = np.concatenate((np.zeros([size,2]),np.ones[size,1]),axis = 1)
    u = project(ed-start)
    O = I - np.einsum('ij,ik-> ijk',u,u)
    first_2coordinates = 2*np.einsum('i,ij -> ij',np.sqrt(X*(1-X)),Y)
    last_ccordinate = 1-2*X
    vec = np.concatenate((first_2coordinates,last_ccordinate),axis = 1)
    return np.einsum('ijk,ik-> ij',O,vec)