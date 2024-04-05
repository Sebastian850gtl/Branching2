# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:18:46 2024

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

import numpy as np
import matplotlib.pyplot as plt

from Brownian import Model

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

np.random.seed(13)
#%% Adjusting dt
# Plots for adjusting the time step


#%% Just reducing time step for small encoutners
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.1 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Rslow = min(r*4,R/4)

sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/4)
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmin_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 4000
if run:
    for n in [1,5,10,20]:
        alpha_max = 1
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max = alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass

# Plots
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Reducing both with Rslow = min(r*4,R/4)
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.1 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Rslow = min(r*4,R/4)
sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/4)
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)


Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmindtmax_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 4000
if run:
    for n in [1,5,10,20]:
        alpha_max = 1/n
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max =alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Reducing both with Rslow = min(r*4,R/8)
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.1 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/8)
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

Rslow = min(r*4,R/8)
Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmindtmax_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 6000
if run:
    for n in [40]:
        alpha_max = 1/n
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max =alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20,40]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Just reducing time step for small encoutners r = 0.05
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.05 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/4)
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

Rslow = min(r*4,R/4)
Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmin_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 4000
if run:
    for n in [1,5,10,20]:
        dtmax = (Rslow/sigma)**2
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max =alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass

# Plots
# Simus with different Rslow
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Reducing both with Rslow = min(r*4,R/4) r= 0.05
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.05 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/4)
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

Rslow = min(r*4,R/4)
Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmindtmax_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 4000
if run:
    for n in [1,5,10,20]:
        alpha_max = 1/n
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max =alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Reducing both with Rslow = min(r*4,R/10)
x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.05 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : min(radius*4,R/10) 
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

Rslow = min(r*4,R/10)
Ntmax = np.inf
print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
file_sim = 'CSdtmindtmax_'+str(Ntmax)+'_'+str(Rslow)+'_'+str(r)+'/'
if not os.path.exists(save_path+file_sim):
    os.makedirs(save_path+file_sim)
run = True
n_sample = 4000
if run:
    for n in [1,5,10,20,40]:
        alpha_max = 1/n
        alpha_min = 1/n
        save_name = save_path+file_sim + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,alpha_max =alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [1,5,10,20,40]:
    save_name = save_path+file_sim + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)
    cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
    plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
plt.legend()
if not os.path.exists(fig_path+file_sim):
    os.makedirs(fig_path+file_sim)
plt.savefig(fig_path+file_sim+'fig.png')

#%% Varrying Rslow
# x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
# r = 0.05 # rayon d'un cluster de taille 1
# sigma = 1
# n_clusters = 2
# sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
# radiusf = lambda x : r
# M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,slow_radius = Rslowf)

# Ntmax = np.inf
# print('dtmax = '+str(2*(Rslow/sigma)**2))
# print('dtmin = '+str(2*(r/sigma)**2))

# # simulation
# file_sim = 'CSRslow_'+str(Ntmax)+'_'+str(r)+'/'
# if not os.path.exists(save_path+file_sim):
#     os.makedirs(save_path+file_sim)
# run = True
# n_sample = 4000
# if run:
#     for n in [1,2,4,5,10]:
#         Rslow = min(r*n,R/2)
#         dtmax = (Rslow/sigma)**2/10
#         dtmin = (r/sigma)**2/10
#         save_name = save_path+file_sim + str(n) 
#         M.run(Ntmax = Ntmax,alpha_min = alpha_max,n_samples = n_sample,save_name = save_name)
# else:
# else:
#     pass
# plt.figure(dpi = 300)
# plt.title('r = '+str(r))
# for n in [1,2,4,5,10]:
#     save_name = save_path+file_sim + str(n) 
#     sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
#     n_sample,_ = sample_times.shape
#     cum_n_sample = np.arange(1,n_sample+1)
#     cumsum_times = np.cumsum(sample_times[:,0])/cum_n_sample
#     plt.plot(cum_n_sample,cumsum_times,label = '$r\times'+str(n)+'$')
# plt.legend()
# if not os.path.exists(fig_path+file_sim):
#     os.makedirs(fig_path+file_sim)
# plt.savefig(fig_path+file_sim+'fig.png')
