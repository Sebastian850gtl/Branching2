import os,sys
#file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

sys.path.append('../lib')

import numpy as np
import matplotlib.pyplot as plt
from concatenator import concatenate_sim
file_name = str(sys.argv[1])

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

#%% Parameters
r = 0.01 # rayon d'un cluster de taille 1
sigma1 = 1
sigma2 = 0
n_clusters = 2
Ntmax = np.inf


# Plots
# Simus with different Rslow
R = 1
Ttheoric = (-np.log(r/R) +np.log(2) -1/2 )* 1/(sigma1**2/2 + sigma2**2/2)

plt.figure(dpi = 300)
#plt.title('r = '+str(r))
for n in [2,3,4,6,10]:#,10,14]:
    tol = 10**(-n)
    save_path_n = save_path +"tol_e-"+ str(n)

    # Concatenate simulations from different runs and load results
    sample_sizes, sample_times = concatenate_sim(save_path_n)
    n_sample,_ = sample_times.shape
    cum_n_sample = np.arange(1,n_sample+1)

    cumsum_times = np.cumsum(sample_times[:,-1])/cum_n_sample
    print("tol = 10^{-%s}, T  = %.2f" %(n,cumsum_times[-1]))

    start = 5
    plt.plot(cum_n_sample[start:],cumsum_times[start:],label = r'tol $= 10^{-%s}, T  = %.2f$' %(n,np.mean(sample_times[:,-1])))
plt.plot(cum_n_sample,Ttheoric*np.ones([n_sample]),label = r"Theoric time $T = %.2f$"%(Ttheoric))
plt.legend()
plt.savefig(fig_path+file_name+'_fig.png')

plt.figure(dpi = 300)
#plt.title('r = '+str(r))
means = []

for n in [2,3,4,6,10]:#,10,14]:
    tol = 10**(-n)
    save_path_n = save_path +"tol_e-"+ str(n)

    # Concatenate simulations from different runs and load results
    sample_sizes, sample_times = concatenate_sim(save_path_n)
    n_sample,_ = sample_times.shape
    means.append(np.mean(sample_times[:,-1]))
print(means)
l = len(means)
plt.scatter([2,3,4,6,10],means,label ='$\mathbb{E}[\hat{T}]$',color = 'blue')
plt.plot([2,3,4,6,10],Ttheoric*np.ones([l]),label = 'Theoric Time',color = 'darkorange')
plt.legend()
plt.savefig(fig_path+file_name+'CV_fig.png')