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
samples = sys.argv[2] # Number of samples 
plot = bool(int(sys.argv[3]))

print(" Simulation of "+str(file_name)+", simulation tag : "+str(runtag))
print(" Note that the tag serves also as a seed")
np.random.seed(int(runtag))
n_sample = int(samples)
#Files locations

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)


#Parameters
r1 = 0.01 # rayon d'un cluster de taille 1
D1 = 1
D2 = 1

sigma1 = np.sqrt(2 * D1)
sigma2 = np.sqrt(2 * D2)

n_clusters = 2
Ntmax = np.inf

sigmaf = lambda x : sigma1 (x<= 1) + sigma2 *(x>1)

#M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
tol = 1/30
radius_range = np.linspace(0.001,0.1,6)
# Simulation

if not plot:
    for i,r2 in enumerate(radius_range):
        radiusf = lambda x : r1 * (x<= 1) + r2 *(x>1)

        M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)

        save_path_i = save_path +"radius_"+ str(i) + '/tmp'
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        save_name = save_path_i +"/simtag_" +runtag 
        M.run(Ntmax = Ntmax,tol = tol,
                    n_samples = n_sample,save_name = save_name,stop = 1,mass_init = np.array([1,2]))
else:
    from concatenator import concatenate_sim
    import matplotlib.pyplot as plt
    # Plots
    #radius_range = np.linspace(0.001,0.1,6)
    meanTs = []
    Terr_95 = []
    for i,r2 in enumerate(radius_range):
        save_path_i = save_path +save_path +"radius_"+ str(i)

        # Concatenating and loading samples
        sample_sizes, sample_times = concatenate_sim(save_path_i)
        n_sample,_ = sample_times.shape

        Tmean = np.mean(sample_times[:,1])
        Tstd = np.std(sample_times[:,1])
        meanTs.append(Tmean)
        Terr_95.append(Tstd* 1.96 / np.sqrt(n_sample))
        print(Tstd, Tmean)

    meanTs = np.array(meanTs)
    Terr_95 = np.array(Terr_95)

    np.save(save_path + "vaues_of_r2.npy",radius_range)
    np.save(save_path + "means.npy",meanTs)
    np.save(save_path + "Terr_95.npy",Terr_95)

    f = lambda x : (-np.log(r1 + x) + np.log(2) )* 2/(sigma1**2 + sigma2**2)
    radius_linspace = np.linspace(radius_range[0],radius_range[-1],200)

    intercept = np.mean(f(radius_range) - meanTs)
    print(intercept) 
    g = lambda x : f(x) - intercept

    plt.figure(dpi = 300)
    plt.plot(radius_linspace,g(radius_linspace),label = r'$r_2 \mapsto \dfrac{-\log\left(%.4f + r_2\right) + \log(2) }{%.1f + %.1f} + %.2f$'%(r1,sigma1**2/2,sigma2**2/2,intercept),
             color = 'darkorange')
    plt.scatter(radius_range,meanTs,label = r'$\hat{T}$',color = 'blue')
    plt.errorbar(radius_range,meanTs, yerr=Terr_95, fmt='o', color='blue', capsize=5)
    plt.legend()
    plt.ylabel("Time")
    plt.xlabel(r"$r_2$")
    plt.savefig(fig_path+file_name+'_fig.png')
    plt.show()