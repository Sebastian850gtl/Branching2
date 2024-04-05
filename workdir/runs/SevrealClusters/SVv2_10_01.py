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
from Brownian import Modelv2 as Model

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


var1 = sys.argv[1]

seed = np.random.randint(10000)
print(seed)
np.random.seed(seed )

runvar = sys.argv[2]


#%% Just reducing time step for small encoutners

radius = 0.01 # rayon d'un cluster de taille 1
sigma = 1
Ntmax = np.inf

tol = 1e-4
sigmaf = lambda x : sigma # diffusion d'un clu
radiusf = lambda x : radius

n_clusters_range = [2,5,10,20,50]
# Simulation
run = bool(int(runvar))
n_sample = 200
if run:
    for i,n_clusters in enumerate([n_clusters_range]):
        monodisperse = np.ones([n_clusters])/n_clusters

        M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)

        save_name = save_path + '_' + str(i) 
        M.run(Ntmax = Ntmax,tol = tol,
                    n_samples = n_sample,save_name = save_name,stop = 1,size_init = monodisperse)
else:
    import matplotlib.pyplot as plt
    # Plots
    # Simus with different Rslow

    
    T_theoric =  -(np.log(radius) +np.log(2)-1)

    for i,n_clusters in enumerate([n_clusters_range]):
        save_name = save_path + '_' + str(i) 
        sample_times =  np.load(save_name+'_times.npy')
        print(sample_times.shape)
        Tmean = np.mean(sample_times[:,1])

        n_sample,_ = sample_times.shape
        print(n_sample)
        cum_n_sample = np.arange(1,n_sample+1)

        cumsum_times = np.cumsum(sample_times[:,-1])/cum_n_sample
        plt.figure(dpi = 300)
        plt.title('MFPT of n Brownian spheres of radius r = '+str(radius)+' with varrying diffusion coef \n on the semi-sphere with reflective boundary conditions.')
        plt.plot(cum_n_sample,2*T_theoric/(n_clusters*(n_clusters - 1)))
        plt.ylabel("Time")
        plt.xlabel("Number of iterations")
        plt.savefig(fig_path+file_name+'_'+str(n_clusters)+'_fig.png')