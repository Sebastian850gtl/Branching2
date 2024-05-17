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
from Brownian import Modelv3 as Model


runtag = sys.argv[1]  # Simulation tag
samples = sys.argv[2] # Number of samples used only if runvar == 1
plot = bool(int(sys.argv[3]))

print(" Simulation of "+str(file_name)+", simulation tag : "+str(runtag))
print(" Note that the tag serves also as a seed")
np.random.seed(int(runtag))
n_sample = int(samples)
#Files locations

save_path = '../../results/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)



#%% Just reducing time step for small encoutners

radius = 0.001 # rayon d'un cluster de taille 1
sigma1 = 1
n_clusters = 2
Ntmax = np.inf

radiusf = lambda x : radius

if not plot:
    # Simulation
    tol = 1/20

    diffusion_range = np.arange(7,11)/5
    for i,D2 in enumerate(diffusion_range):
        sigma2 = np.sqrt(2*D2)
        sigmaf = lambda x : sigma1*(x<= 1) + sigma2*(x > 1)
        M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
        save_path_i = save_path +"diffusion_"+ str(i) + '/tmp'
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        save_name = save_path_i +"/simtag_" +runtag 
        M.run(Ntmax = Ntmax,tol = tol,
                    n_samples = n_sample,save_name = save_name,stop = 1,size_init = np.array([2,1]))
        
else:
    from concatenator import concatenate_sim
    import matplotlib.pyplot as plt
    diffusion_range = np.arange(11)/5

    # Plots
    # Simus with different Rslow

    D1 = sigma1**2/2
    R = 1
    f = lambda x : R**2 *(-np.log(2*radius/R) +np.log(2) )/(x + D1) 


    diff_linspace = np.linspace(diffusion_range[0],diffusion_range[-1]*1.05,200)

    meanTs = []
    diff_range_plot =  diffusion_range
    for i,D2 in enumerate(diffusion_range):
        sigma2 = np.sqrt(2*D2)
        save_path_i = save_path +"diffusion_"+ str(i)

        # Concatenating and loading samples
        sample_sizes, sample_times = concatenate_sim(save_path_i)
        n_sample,_ = sample_times.shape
        
        Tmean = np.mean(sample_times[:,1])
        meanTs.append(Tmean)
    meanTs = np.array(meanTs)


    plt.figure(dpi = 300)
    #plt.title('MFPT of 2 Brownian spheres of radius r = '+str(radius)+' with varrying diffusion coef \n on the semi-sphere with reflective boundary conditions.')
    plt.plot(diff_linspace,f(diff_linspace),label = r'$D_B \mapsto \dfrac{-\log\left(\dfrac{r_A + r_B}{R}\right) + \log(2) }{D_B + D_A}$',color = 'darkorange')
    plt.scatter(diff_range_plot,meanTs,label = r'$\hat{T}$',color = 'blue')
    plt.legend()
    plt.ylabel("Time")
    plt.xlabel(r"$D_B$")
    plt.savefig(fig_path+file_name+'_fig.png')