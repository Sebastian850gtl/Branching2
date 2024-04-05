# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:37:15 2024

@author: sebas
"""

import os,sys
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import numpy as np
from ML import Coalescence

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

#Parameters
# ML run with parameters that relate to the Brownina run

n_clusters = 100

R = 1 # Complete radius of the sphere
radius_0 = 0.001 # radius of a cluster that has mass 1/N0
D0 = 1 # Diffusion coefficient that has mass 1/N0

alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/3,1/2,1]
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
# Simulation
run = bool(int(runvar))
n_sample = 7000
if run:
    for i,alpha in enumerate(alpha_range):
        for j,beta in enumerate(beta_range):
            print(alpha,beta)
            
            kernel = lambda x,y : (1/x**alpha + 1/y**alpha) * 1/(-R**2*np.log(radius_0/R*((n_clusters*x)**beta+(n_clusters*y)**beta)) + 1)
            M = Coalescence(n_clusters = n_clusters,kernel = kernel)

            save_name = save_path + '_' + str(i) + '_' +str(j)
            M.run(n_samples = n_sample, init = monodisperse, save_name = save_name)
else:
    import matplotlib.pyplot as plt
    from compute_probas import probs
    # Plots
    # Simus with different Rslow

    for i,alpha in enumerate(alpha_range):
        plt.figure(dpi = 300)
        plt.title(r"ML Branching probabability for $\alpha = %.2f , r_0 = %.3f$ \n and $N_0 = %s $"% (alpha,radius_0,n_clusters))
        for j,beta in enumerate(beta_range):
            save_name = save_path + '_' + str(i) + '_' +str(j)
            sample_times =  np.load(save_name+'_times.npy')
            sample_sizes =  np.load(save_name+'_sizes.npy')
            n_samples,n_clusters = sample_times.shape
        
            time_range = np.linspace(0,3*np.mean(sample_times[:,-1]),200)
            print("Computing probas, parameters : alpha, beta =" + str((alpha,beta)))
            probies = probs(sample_sizes,sample_times,time_range,2,0.4)

            #plt.figure(dpi = 300)
            #plt.title('ML Branching probabability for alpha = '+str(alpha)+', beta ='+str(beta))
            plt.plot(time_range,probies, label = r"$\beta = %.2f $"%(beta))
        plt.legend()
        plt.savefig(fig_path+file_name+'_'+str(i)+'_fig.png')