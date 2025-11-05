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
from scipy.stats import norm
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

#%% Parameters
r = 0.001 # rayon d'un cluster de taille 1
sigma1 = 1
sigma2 = 0
n_clusters = 2
Ntmax = np.inf

sigmaf = lambda x : sigma1*(x<= 1) + sigma2*(x>1)
radiusf = lambda x : r

M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)


# simulation
if not plot:
    M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
    for n in [5,10,20,40]:
        tol = 1/n
        save_path_n = save_path +"tol_"+ str(n)+'/tmp'
        if not os.path.exists(save_path_n):
            os.makedirs(save_path_n)
        save_name = save_path_n +"/simtag_" +runtag
        M.run(Ntmax = Ntmax,tol = tol,
                    n_samples = n_sample,save_name = save_name,stop = 1,mass_init = np.array([2,1]),reduced = True)

else:
    import matplotlib.pyplot as plt
    from concatenator import concatenate_sim

    save_path = '../../results/'+file_name+'/'
    fig_path = '../../results/fig/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    #%% Parameters
    r = 0.001 # rayon d'un cluster de taille 1
    sigma1 = 1
    sigma2 = 0
    n_clusters = 2
    Ntmax = np.inf


    # Plots
    # Simus with different Rslow
    R = 1
    Ttheoric = (-np.log(2*r/R) + np.log(2))* 1/(sigma1**2/2 + sigma2**2/2)

    plt.figure(dpi = 300)
    plt.xlabel("Number of samples")
    for n in [10,20,40]:#,50]:
        tol = 1/n
        save_path_n = save_path +"tol_"+ str(n)
        
        # Concatenate simulations from different runs and load results
        sample_sizes, sample_times = concatenate_sim(save_path_n)

        n_sample,_ = sample_times.shape
        print("Samplu", n_sample)
        cum_n_sample = np.arange(2,n_sample+1)


        cum_sum_x = np.cumsum(sample_times[1:,-1])
        cum_sum_x_2 = np.cumsum(sample_times[1:,-1]**2)

        cumsum_times = cum_sum_x/cum_n_sample
        print("tol = 1/%s, T  = %.2f" %(n,cumsum_times[-1]))
        print("prob = %.2f" %(np.exp(-n)))

        sample_var = 1 /(cum_n_sample-1) *(cum_sum_x_2 - cum_sum_x**2 / cum_n_sample)

        std_times = np.sqrt(sample_var / cum_n_sample)
        start = 1000
        end = n_sample

        label = r" tol$ = (%s)^{-1}, \hat{{\tau}}_{{AB}}  = %.2f$" % (n, np.mean(sample_times[:, -1]))
        line, = plt.plot(cum_n_sample[start:end], cumsum_times[start:end], label=label)

        color = line.get_color()
        ci = 1.96 * std_times[start:end]  # 95% confidence interval

        plt.fill_between(cum_n_sample[start:end],
                     cumsum_times[start:end] - ci,
                     cumsum_times[start:end] + ci,
                     color=color, alpha=0.2)
    
        #plt.plot(cum_n_sample[start:end], Ttheoric * np.ones(end - start),
        # label=r"Theoric time $T = %.2f$" % Ttheoric)
    

    plt.plot(cum_n_sample[start:end],Ttheoric*np.ones([end - start - 1]),label = r"Theoric time $T = %.2f$"%(Ttheoric))
    plt.legend()
    plt.savefig(fig_path+file_name+'_fig.png')
    plt.show()
