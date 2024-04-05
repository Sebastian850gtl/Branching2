import os,sys
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
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
sigma = 1
n_clusters = 2
Ntmax = np.inf

n = 29
alpha_max = 1/n
alpha_min = 1/n

sigmaf = lambda x : sigma*(x<= 1) # diffusion d'un cluster d'une seule particule egale a 0 pour une particule plus grande que 1 (pour que le cluster au centre soit immobile)
radius_range = [0.002,0.004,0.008,0.012,0.02,0.04,0.1]
# simulation
run = bool(int(runvar))
n_sample = 100
if run:
    for r in radius_range:
        print(" Simulation for radius = " +str(r))
        radiusf = lambda x : r
        Rslow = min(np.sqrt(r*R),0.2)
        Rslowf = lambda radius : Rslow
        M = Model(tip_param,n_clusters = n_clusters,sigma = sigmaf,radius = radiusf,
          slow_radius = Rslowf)

        save_name = save_path + '_'+str(r) 
        M.run(Ntmax = Ntmax,alpha_min = alpha_min,
              alpha_max = alpha_max,n_samples = n_sample,save_name = save_name,position_init = 'center',size_init = np.array([2,1]))
else:
    pass

# Plots
# Simus with different Rslow
plt.figure(dpi = 300)
plt.title('Fitting')
meanTs = []
for r in radius_range:
    save_name = save_path + '_'+str(r) 
    sample_times =  np.load(save_name+'_times.npy')
    print(sample_times.shape)
    Tmean = np.mean(sample_times[:,1])
    meanTs.append(Tmean)
    
    rlinsapce  = np.linspace(radius_range[0],radius_range[-1],200)
meanTs = np.array(meanTs)

fun = lambda x : 2*(-np.log(x) -2*x**2 + 1/2 *np.log(1-x**2) + 1)
A = np.vstack([fun(np.array(radius_range)), np.ones(len(radius_range))]).T
a, intercept = np.linalg.lstsq(A, meanTs, rcond=None)[0]
print(intercept,a)
print(intercept/a)
ffit = lambda x : a*fun(x) + intercept

intercept2 = np.sqrt(np.mean((meanTs - fun(np.array(radius_range)))**2))
ffit2 = lambda x : fun(x) - intercept2
print(intercept2)
plt.plot(rlinsapce,ffit(rlinsapce))
plt.plot(rlinsapce,ffit2(rlinsapce))
plt.scatter(radius_range,meanTs)

plt.savefig(fig_path+file_name+'_fig.png')