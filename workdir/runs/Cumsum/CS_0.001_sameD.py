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
from Brownian import Modelv2 as Model


runtag = sys.argv[1]  # Simulation tag
samples = sys.argv[2] # Number of samples used only if runvar == 1

print(" Simulation of "+str(file_name)+", simulation tag : "+str(runtag))
print(" Note that the tag serves also as a seed")
#np.random.seed(int(runtag))
n_sample = int(samples)
#Files locations

save_path = '../../results/'+file_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#%% Parameters
r = 0.001 # rayon d'un cluster de taille 1
sigma = 1
n_clusters = 2
Ntmax = np.inf

sigmaf = lambda x : sigma # diffusion d'un clu
radiusf = lambda x : r

M = Model(n_clusters = n_clusters,sigmafun = sigmaf,radfun = radiusf)


# simulation

for n in [2,3,4,6,10]:
    tol = 10**(-n)
    print( 'keq =' +str(1/ norm.ppf(tol)**(-2)) )
    save_path_n = save_path +"tol_e-"+ str(n)+'/tmp'
    if not os.path.exists(save_path_n):
        os.makedirs(save_path_n)
    save_name = save_path_n +"/simtag_" +runtag
    
    M.run(Ntmax = Ntmax,tol = 10**(-n),
                n_samples = n_sample,save_name = save_name,stop = 1,size_init = np.array([1,1]))