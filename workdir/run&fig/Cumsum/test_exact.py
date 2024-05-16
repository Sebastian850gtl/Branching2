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
from Brownian import reflected_brownian_sphere, reflected_brownian_sphere_old,sample_WF,sampleA_infinte_exact
from time import time


np.random.seed(13)
n_samples = int(sys.argv[1])
t = float(sys.argv[2])

start = np.array([0,1,0]).reshape(-1,3)
sigmas = np.array([1])
S_exact = []
S_approx = []
X_WF = []
sample_inf = []
print("t is :"+str(t))

# t10 = time()
# for k in range(n_samples):
#     if 10*k %n_samples == 0:
#         print("Step "+str((10*k)//n_samples))  
#     S_approx.append(reflected_brownian_sphere_old(start,sigmas,t,radiuses = 0))
# print(time() - t10)
# t20 = time()
# for k in range(n_samples):
#     if 10*k %n_samples == 0:
#         print("Step "+str((10*k)//n_samples))  
#     S_exact.append(reflected_brownian_sphere(start,sigmas,t,radiuses = 0,switch = t*0.5))
# print(time() - t20)

for k in range(n_samples):
    if 10*k %n_samples == 0:
        print("Step "+str((10*k)//n_samples))  
    
    X_WF.append(sample_WF(t*sigmas,1,1))
    sample_inf.append(sampleA_infinte_exact(t))
# print(np.mean(S_approx))
# print(np.mean(S_exact))
print(np.mean(X_WF))
print(np.mean(sample_inf))