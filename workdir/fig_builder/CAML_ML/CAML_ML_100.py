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
from CAML import CAML

runtag = sys.argv[1]  # Simulation tag
samples = int(sys.argv[2]) # Number of samples used only if runvar == 1
plot = bool(int(sys.argv[3]))

np.random.seed(int(runtag))
n_sample = samples
#Files locations

save_path = '../../results/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

save_path = save_path
#Parameters
n_clusters = 100
alpha_range = [0,0.25,0.5,0.75,1]
# Initial distribution 
monodisperse = np.ones([n_clusters])/n_clusters

if not plot:
    # Simulation
    for i,alpha in enumerate(alpha_range):
        # print(" Running ML")
        # kernel = lambda x,y : (1/x**alpha + 1/y**alpha) 
        # M1 = Coalescence(n_clusters = n_clusters,kernel = kernel)
        # save_path_i_ML = save_path + 'ML_' + str(i) + '/tmp'
        # if not os.path.exists(save_path_i_ML):
        #     os.makedirs(save_path_i_ML)
        # save_name = save_path_i_ML + "/simtag_" +runtag 
        # M1.run(n_samples = n_sample, init = monodisperse, save_name = save_name)

        print(" Running CAML")
        M2 = CAML(n_clusters = n_clusters,alpha = alpha)
        save_path_i_CAML = save_path + 'CAML_' + str(i) + '/tmp'
        if not os.path.exists(save_path_i_CAML):
            os.makedirs(save_path_i_CAML)
        save_name = save_path_i_CAML + "/simtag_" +runtag 
        M2.run(n_samples = n_sample, init = monodisperse, save_name = save_name)
else:
    # Plots
    from compute_probas import probs
    from concatenator import concatenate_sim
    import matplotlib.pyplot as plt

    for i,alpha in enumerate(alpha_range):
        plt.figure(dpi = 300)
        #plt.title(r"ML vs CAML Branching probabability for $\alpha = %.2f , r_0 = %.3f$ \n and $N_0 = %s $"% (alpha,radius_0,n_clusters))
        
        save_path_i_ML = save_path + 'ML_' + str(i) 
        sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
        n_samples,n_clusters = sample_times_ML.shape

        time_range = np.linspace(0,np.max(sample_times_ML[:,-1]),200)
        print("Computing probas for ML, parameters : alpha =" + str((alpha)))
        print("Number of samples :"+str(n_samples))

        #print(len(np.where(sample_sizes[-1,:,:] > 0)[0])/n_samples)
        probies = probs(sample_sizes_ML,sample_times_ML,time_range,2,0.3)
        plt.plot(time_range,probies, label = r"ML")

        save_path_i_CAML = save_path + 'CAML_' + str(i) 
        sample_sizes_CAML, sample_times_CAML = concatenate_sim(save_path_i_CAML)
        n_samples,n_clusters = sample_times_CAML.shape

        time_range = np.linspace(0,np.max(sample_times_CAML[:,-1]),200)
        print("Computing probas for CAML, parameters : alpha =" + str((alpha)))
        print("Number of samples :"+str(n_samples))

        #print(len(np.where(sample_sizes[-1,:,:] > 0)[0])/n_samples)
        probies = probs(sample_sizes_CAML,sample_times_CAML,time_range,2,0.3)
        plt.plot(time_range,probies, label = r"CAML")

        plt.legend()
        plt.savefig(fig_path+file_name+'_'+str(i)+'_fig.png')