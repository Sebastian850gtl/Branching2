# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:13:57 2024

@author: sebas
"""

import os,sys
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from Brownian import Model

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

var1 = sys.argv[1]
print(var1)
np.random.seed(int(var1))
runvar = sys.argv[2]
#%% Reducing both time steps

x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
sigma = 1
n_clusters = 2
Rslow = 0.1
Rslowf = lambda radius : Rslow
Ntmax = np.inf

n = 10
alpha_max = 1/n
alpha_min = 1/n

sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule

radius_range = [0.005,0.01,0.02,0.04,0.06,0.1]
# simulation
run = bool(int(runvar))
n_sample = 10
if run:
    for r in radius_range:
        print(" Simulation for radius = " +str(r))
        radiusf = lambda x : r
        M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,
          slow_radius = Rslowf)

        save_name = save_path + '_'+str(r) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,
              alpha_max = alpha_max,n_samples = n_sample,save_name = save_name)
else:
    pass

# Plots
# Simus with different Rslow
plt.figure(dpi = 300)
plt.title('Fitting')
meanTs = []
for r in radius_range:
    save_name = save_path + '_'+str(r) 
    sample_times =  np.load(save_name+'_times.npy')
    print(sample_times.shape)
    Tmean = np.mean(sample_times[:,1])
    meanTs.append(Tmean)
    
    rlinsapce  = np.linspace(radius_range[0],radius_range[-1],200)
meanTs = np.array(meanTs)
A = np.vstack([-np.log(radius_range), np.ones(len(radius_range))]).T
a, intercept = np.linalg.lstsq(A, meanTs, rcond=None)[0]
print(a,intercept)
ffit = lambda x : -a*np.log(x) + intercept
plt.plot(rlinsapce,ffit(rlinsapce))
plt.scatter(radius_range,meanTs)

plt.savefig(fig_path+file_name+'_fig.png')