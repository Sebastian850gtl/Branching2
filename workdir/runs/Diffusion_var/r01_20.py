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
from Brownian import Model,uniform_init

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

x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
sigma1 = 1
n_clusters = 2
Ntmax = np.inf
radius = 0.1
n = 20
alpha_max = 1/n
alpha_min = 1/n

radiusf = lambda x : radius
Rslow = min(np.sqrt(radius*R),0.2)
Rslowf = lambda radius : Rslow

diffusion_range = np.arange(15)/2
# Simulation
run = bool(int(runvar))
n_sample = 1000
if run:
    for i,D2 in enumerate(diffusion_range):
        sigma2 = np.sqrt(2*D2)
        sigmaf = lambda x : sigma1*(x<= 1) + sigma2*(x > 1)
        M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,
          slow_radius = Rslowf)

        save_name = save_path + '_' + str(i) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,
              alpha_max = alpha_max,n_samples = n_sample,save_name = save_name,stop = n_clusters-1,size_init = np.array([2,1]))
else:
    import matplotlib.pyplot as plt
    # Plots
    # Simus with different Rslow

    D1 = sigma1**2/2
    f = lambda x : -(np.log(radius/R) +np.log(2)-1)* 1/(x + D1)
    diff_linspace = np.linspace(diffusion_range[0],diffusion_range[-1]*1.05,200)

    meanTs = []
    diff_range_plot =  diffusion_range
    for i,D2 in enumerate(diffusion_range):
        sigma2 = np.sqrt(2*D2)
        save_name = save_path + '_' + str(i) 
        sample_times =  np.load(save_name+'_times.npy')
        print(sample_times.shape)
        Tmean = np.mean(sample_times[:,1])
        meanTs.append(Tmean)
    meanTs = np.array(meanTs)

    
    plt.figure(dpi = 300)
    plt.title('MFPT of 2 Brownian spheres of radius r = '+str(radius)+' with varrying diffusion coef \n on the semi-sphere with reflective boundary conditions.')
    plt.plot(diff_linspace,f(diff_linspace),label = '$D \mapsto \dfrac{-\log(r/R)-\log(2)+1}{D + 0.5}$',color = 'darkorange')
    plt.scatter(diff_range_plot,meanTs,label = '$\mathbb{E}[\hat{T}]$',color = 'blue')
    plt.legend()
    plt.ylabel("Time")
    plt.xlabel("Diffusion coeficient D")
    plt.savefig(fig_path+file_name+'_fig.png')