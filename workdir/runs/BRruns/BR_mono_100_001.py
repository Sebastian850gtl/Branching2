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

np.random.seed(int(runtag))
n_sample = int(samples)
#Files locations

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#Parameters

n_clusters = 100
Ntmax = np.inf
radius_0 = 0.001 # radius of a cluster that has mass 1/N0
D0 = 1
# Parameters of time step scaling
tol = 1/30

# range of paramters for the radius and diffusion function
alpha_range = [0,1/3,2/3,1]
beta_range = [0,1/2] 
# Initial distribution

monodisperse = np.ones([n_clusters])/n_clusters
if not plot:
    for i,alpha in enumerate(alpha_range):
        for j,beta in enumerate(beta_range):
            print(alpha,beta)
            radiusf = lambda x : radius_0 * (n_clusters*x)**(beta)
            sigmaf = lambda x : np.sqrt(2*D0*(n_clusters*x)**(-alpha))
            
            M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
            save_path_n = save_path +"alpha_beta_"+  str(i) + '_' +str(j)+'/tmp'
            if not os.path.exists(save_path_n):
                os.makedirs(save_path_n)
            save_name = save_path_n +"/simtag_" +runtag
            M.run(Ntmax = Ntmax,tol = tol,
                    n_samples = n_sample,save_name = save_name,stop = 1,size_init = monodisperse)
        
else: 

    import matplotlib.pyplot as plt
    from compute_probas import probs
    from concatenator import concatenate_sim

    for i,alpha in enumerate(alpha_range):
        plt.figure(dpi = 300)
        plt.title(r"ML Branching probabability for $\alpha = %.2f , r_0 = %.3f$ \n and $N_0 = %s $"% (alpha,radius_0,n_clusters))
        for j,beta in enumerate(beta_range):
            save_path_n = save_path +"alpha_beta_"+  str(i) + '_' +str(j)
            sample_sizes, sample_times = concatenate_sim(save_path_n)
            n_samples,n_clusters = sample_times.shape

            time_range = np.linspace(0,np.max(sample_times[:,-1]),200)
            print("Computing probas, parameters : alpha, beta =" + str((alpha,beta)))
            print("Number of samples :"+str(n_samples))

            print(len(np.where(sample_sizes[-1,:,:] > 0)[0])/n_samples)
            probies = probs(sample_sizes,sample_times,time_range,2,0.3)

            #plt.figure(dpi = 300)
            #plt.title('ML Branching probabability for alpha = '+str(alpha)+', beta ='+str(beta))
            plt.plot(time_range,probies, label = r"$\beta = %.2f $"%(beta))
        plt.legend()
        plt.savefig(fig_path+file_name+'_'+str(i)+'_fig.png')