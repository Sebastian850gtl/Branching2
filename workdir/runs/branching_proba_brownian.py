# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:47:53 2024

@author: sebas
"""

import os,sys
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')
wd = os.getcwd() # defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

from Brownian import Model
from compute_probas_brownian import _sort_, xk_greater_than

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

np.random.seed(13)
#%% Constant sigma and radius
runmodel = True # Wether we run the model or just load the results 

tip_param = np.array([0,0,0]),1,0
x0,R,phi0 = tip_param  # respectivement le centre de la sphere, son rayon et l'azimuth du cercle delimitant le domaine (azimuth = 0 demi-sphere , azimuth = -pi/2 sphere entière)
r = 0.01 # rayon d'un cluster de taille 1
sigma = 1

#dx = 2*r # distance minimale entre deux centre de cluster pour une collision (pas au point)
sigmaf = lambda x : 1 # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r

#dt = min(0.01,(dx/sigma)**2/3) # pas adapté à la distance minimale (pas au point)
dt = 0.05 # pas de temps

Ntmax = 2000
N = 50 # Nombres de points ils seront placés uniformément sur le domaine.
n_samples = 5000

save_file = save_path+'constant_'+str(sigma)+'_'+str(r)+'_'+str(Ntmax)+'_'+str(dt)+'_'+'N'

if runmodel:
    M = Model(tip_param,N_points = N,sigma = sigmaf,radius = radiusf)
    M.run(Ntmax=Ntmax,n_samples = n_samples,dt =dt,save_trajectories = False,save_size_file = save_file)
    sizes = M.sizes
    print(len(sizes[-1]))
    
# loading 
with open(save_file, "rb") as fp:
    sample_sizes = pickle.load(fp)
_sort_(sample_sizes)
# Computing probas
Probies = xk_greater_than(sample_sizes,2,0.4*N,Ntmax+2)

#figure
save_fig_file = save_path + 'constant_'+str(sigma)+'_'+str(r)+'_'+str(Ntmax)+'_'+str(dt)+'_'+'N'+'.png'
t = np.linspace(0,dt*Ntmax,Ntmax+2)
plt.figure(dpi = 300)
plt.plot(t,Probies)
plt.savefig(save_fig_file)
