# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:34:03 2024

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
from Brownian import Model

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

var1 = sys.argv[1]
np.random.seed(int(var1))
runvar = sys.argv[2]
#%% Reducing both time steps

x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.1 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Rslow = 0.1
Ntmax = np.inf

sigmaf = lambda x : sigma # diffusion d'un cluster d'une seule particule
radiusf = lambda x : r
Rslowf = lambda radius : Rslow
M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,
          slow_radius = Rslowf)

print('dtmax = '+str((Rslow/sigma)**2))
print('dtmin = '+str((r/sigma)**2))

# simulation
run = bool(int(runvar))
n_sample = 200
if run:
    for n in [1,2,5,10]:
        alpha_max = 1/n
        alpha_min = 1/n
        save_name = save_path + str(n) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,
              alpha_max = alpha_max,n_samples = n_sample,save_name = save_name)
else:
    import matplotlib.pyplot as plt
    from scipy.stats import kstest
    # Plots
    # Simus with different Rslow
    plt.figure(dpi = 300)
    plt.title('r = '+str(r))
    for n in [1,2]:
        save_name = save_path + str(n) 
        sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
        n_sample,_ = sample_times.shape
        print(n_sample)
        cum_n_sample = np.arange(1,n_sample+1)
        cumsum_times = np.cumsum(sample_times[:,1])/cum_n_sample
        print('n = '+str(n)+' T = '+str(cumsum_times[-1]))
        plt.plot(cum_n_sample,cumsum_times,label = 'dt/'+str(n))
    plt.legend()
    plt.savefig(fig_path+file_name+'_fig.png')


# plt.figure(dpi = 300)
# plt.title("Histo")
# for n in [1,5,10,20,40]:
#     save_name = save_path + str(n) 
#     sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
#     lbd = 1/np.mean(sample_times[:,0])
#     cdf = lambda x : 1-np.exp(-lbd*x)
#     samples = sample_times[:,0]
#     print(kstest(samples,cdf))
#     n_sample,_ = sample_times.shape
#     plt.hist(sample_times[:,0],bins = n_sample//100,density=True)

# plt.savefig(fig_path+'hist.png')