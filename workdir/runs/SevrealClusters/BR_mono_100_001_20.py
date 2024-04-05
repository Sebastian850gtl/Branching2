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

#Parameters


x0,R,phi0 = tip_param = np.array([0,0,0]),1,0
n_clusters = 100
Ntmax = np.inf
radius_0 = 0.001 # radius of a cluster that has mass 1/N0
D0 = 1

# Parameters of time step scaling
nscaledt = 20 
alpha_max = 1/nscaledt
alpha_min = 1/nscaledt


alpha_range = list(np.linspace(0,1,4))
beta_range = [1/3,5/12] + list(np.linspace(1/2,2,3)) 
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
# Simulation
run = bool(int(runvar))
n_sample = 100
if run:
    for i,alpha in enumerate(alpha_range):
        for j,beta in enumerate(beta_range):

            radiusf = lambda x : radius_0 * (n_clusters*x)**(alpha)
            sigmaf = lambda x : np.sqrt(2*D0*n_clusters*x**(-alpha))
            Rslowf = lambda radius : min(radius * 10,0.05)
            
            M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,
            slow_radius = Rslowf)

            save_name = save_path + '_' + str(i) + '_' +str(j)
            M.run(Ntmax = Ntmax,alpha_min = alpha_min,
                alpha_max = alpha_max,n_samples = n_sample,save_name = save_name,stop = 1,size_init = monodisperse)
else:
    import matplotlib.pyplot as plt
    from compute_probas import probs
    # Plots
    # Simus with different Rslow

    for i,alpha in enumerate(alpha_range):
        for j,beta in enumerate(beta_range):
            save_name = save_path + '_' + str(i) + '_' +str(j)
            sample_times =  np.load(save_name+'_times.npy')
            sample_sizes =  np.load(save_name+'_sizes.npy')

            time_range = np.linsapce(0,5,200)
            print("Computing probas, parameters : $\alpha,\beta =$" + str((alpha,beta)))
            probies = probs(sample_sizes,sample_times,time_range,2)

            plt.figure(dpi = 300)
            plt.title('Brownian Branching probabability for $\alpha = '+str(alpha)+', \beta ='+str(beta)+'$')
            plt.plot(time_range,probies)
            plt.savefig(fig_path+file_name+'_fig.png')