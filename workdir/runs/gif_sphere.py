# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 03:39:05 2022

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

gif_tag = '2'
fig_path = '../../results/fig/'+file_name+str(gif_tag)+'/' # destination du gif
if not os.path.exists(fig_path):
        os.makedirs(fig_path)

import numpy as np
import matplotlib.pyplot as plt
from time import time

from Brownian import Model

os.chdir(fig_path)

#np.random.seed(15)

#%% Model parameters
tip_param = np.array([0,0,0]),1,0
x0,R,phi0 = tip_param  # respectivement le centre de la sphere, son rayon et l'azimuth du cercle delimitant le domaine (azimuth = 0 demi-sphere , azimuth = -pi/2 sphere entière)
r = 0.1 # rayon d'un cluster de taille 1

#dx = 2*r # distance minimale entre deux centre de cluster pour une collision (pas au point)
sigmaf = lambda x : 1 # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r

#dt = min(0.01,(dx/sigma)**2/3) # pas adapté à la distance minimale (pas au point)
dt = 0.005 # pas de temps

Ntmax = 1000
N = 20 # Nombres de points ils seront placés uniformément sur le domaine.
#%% Model running



M = Model(tip_param,N_points = N,sigma = sigmaf,radius = radiusf)
M.run(Ntmax,dt,save_trajectories = True)
trajectories = M.trajectories
sizes = M.sizes

print("Number of time iterations : " + str(len(trajectories)-1))
#%% Making Gif

n_theta = 100 # number of values for theta
n_phi = 300  # number of values for phi
theta, phi = np.mgrid[phi0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]

x = R*np.sin(theta)*np.cos(phi)
y = R*np.sin(theta)*np.sin(phi)
z = R*np.cos(theta)
    
filenames = []

max_iter = 100
iters = min(max_iter,len(trajectories))
skip = 1#len(trajectories)//iters
trail_size = 4
t0 = time()
idi = 0
for index, tra in enumerate(trajectories):
    print('\r',   'Advancement : %.1f'%((idi/iters)*100)+' %', 'done in %.2fs.' % (time() - t0),end='')
    # plot charts
    if idi > iters:
        break
    if index%skip == 0:
        fig = plt.figure(dpi = 30)
        ax = fig.add_subplot(111, projection='3d')
        #axisEqual3D(ax)
        ax.plot_surface(
            x,y,z,  rstride=1, cstride=1, alpha=0.1, linewidth=1)
        
        area = [s*r**2*400 for s in sizes[index]]
        ax.scatter(tra[:,0],tra[:,1],tra[:,2],s = area,c = 'black')
        # create file name and append it to a list
        filename = '%04d'%(idi)+'.png'
        filenames.append(filename)
        
                
        # save frame
        plt.savefig(filename)
        plt.close()
        idi += 1
# build gif
import imageio
with imageio.get_writer('mygifbeg.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
# for filename in set(filenames):
    
#     os.remove(filename)

os.chdir(wd)