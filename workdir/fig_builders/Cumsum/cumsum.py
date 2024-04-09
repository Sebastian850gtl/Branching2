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

file_name = sys.argv[1]

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

#%% Parameters
r = 0.01 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Ntmax = np.inf

# Plots
# Simus with different Rslow
plt.figure(dpi = 300)
plt.title('r = '+str(r))
for n in [2,3,4,6,10]:
    save_name = save_path + str(n) 
    sample_sizes, sample_times = np.load(save_name+'_sizes.npy'), np.load(save_name+'_times.npy')

    n_sample,_ = sample_times.shape
    print(n_sample)
    cum_n_sample = np.arange(1,n_sample+1)

    cumsum_times = np.cumsum(sample_times[:,-1])/cum_n_sample
    print("tol = 10^{-%s}, T  = %.2f" %(n,cumsum_times[-1]))

    start = 5
    plt.plot(cum_n_sample[start:],cumsum_times[start:],label = r'tol $= 10^{-%s}, T  = %.2f$' %(n,np.mean(sample_times[:,-1])))
plt.legend()
plt.savefig(fig_path+file_name+'_fig.png')