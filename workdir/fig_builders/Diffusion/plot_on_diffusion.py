import os,sys
#file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

sys.path.append('../lib')
from concatenator import concatenate_sim
import numpy as np
import matplotlib.pyplot as plt

file_name = sys.argv[1]

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# Parameters
radius = 0.001 # rayon d'un cluster de taille 1
sigma1 = 1
n_clusters = 2
Ntmax = np.inf

radiusf = lambda x : radius

#M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)
tol = 1e-5

diffusion_range = np.arange(11)/5

# Plots
# Simus with different Rslow

D1 = sigma1**2/2
R = 1
f = lambda x : (-np.log(radius/R)  - 1/2)* 1/(x + D1)


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
plt.title('MFPT of 2 Brownian spheres of radius r = '+str(radius)+' with varrying diffusion coef \n on the semi-sphere with reflective boundary conditions.')
plt.plot(diff_linspace,f(diff_linspace),label = '$D \mapsto \dfrac{-\log(r/R)-\log(2)+1}{D + 0.5}$',color = 'darkorange')
plt.scatter(diff_range_plot,meanTs,label = '$\mathbb{E}[\hat{T}]$',color = 'blue')
plt.legend()
plt.ylabel("Time")
plt.xlabel("Diffusion coeficient D")
plt.savefig(fig_path+file_name+'_fig.png')