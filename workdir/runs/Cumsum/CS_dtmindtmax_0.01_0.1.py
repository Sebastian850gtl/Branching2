# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:47:10 2024

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
from Brownian import Modelv2 as Model


save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


seed = np.random.randint(10000)
print(seed)
np.random.seed(seed )

runvar = sys.argv[1]


#%% Just reducing time step for small encoutners

x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
r = 0.01 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Ntmax = np.inf

sigmaf = lambda x : sigma # diffusion d'un clu
radiusf = lambda x : r

M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)


# simulation
run = bool(int(runvar))
n_sample = 1000

from scipy.stats import norm
if run:
    for n in [2,3,4,6,10]:
        tol = 10**(-n)
        print( 'alpha =' +str( norm.ppf(tol)**(-2)) )
        save_name = save_path + str(n) 
        
        M.run(Ntmax = Ntmax,tol = 10**(-n),
                  n_samples = n_sample,save_name = save_name,stop = 1,size_init = np.array([1,1]))
else:
    # Plots
    # Simus with different Rslow
    plt.figure(dpi = 300)
    plt.title('r = '+str(r))
    for n in [2,3,4,6,10]:
        save_name = save_path + str(n) 
        sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')

        #sample_times = sample_times[32000:,:]

        n_sample,_ = sample_times.shape
        print(n_sample)
        cum_n_sample = np.arange(1,n_sample+1)

        cumsum_times = np.cumsum(sample_times[:,-1])/cum_n_sample
        print("tol = 10^{-%s}, T  = %.2f" %(n,cumsum_times[-1]))

        start = 5
        plt.plot(cum_n_sample[start:],cumsum_times[start:],label = r'tol $= 10^{-%s}, T  = %.2f$' %(n,np.mean(sample_times[:,-1])))
    plt.legend()
    plt.savefig(fig_path+file_name+'_fig.png')

    #plt.figure(dpi = 300)
    #plt.title("Histo")
    #for n in [1,5,10,20,40]:
    #    save_name = save_path + str(n) 
    #    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')
    #    lbd = 1/np.mean(sample_times[:,0])
    #    cdf = lambda x : 1-np.exp(-lbd*x)
    #    samples = sample_times[:,0]
    #    print(kstest(samples,cdf))
    #    n_sample,_ = sample_times.shape
    #    plt.hist(sample_times[:,0],bins = n_sample//100,density=True)

    #plt.savefig(fig_path+file_name+'_hist.png')