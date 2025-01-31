import os,sys
import importlib
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('../..')
wd = os.getcwd() # Defines the working directory

sys.path.append('../models')
sys.path.append('../lib')

import numpy as np

#Files locations


fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

parameters_file_name = sys.argv[1]

#Parameters

param_module = importlib.import_module(parameters_file_name)

alpha_range = param_module.alpha_range
x = param_module.x
k = param_module.k
k = 2

SC_folder = param_module.SC_folder
CASC_folder = param_module.CASC_folder

save_path_SC = '../../results/' + SC_folder + '/'
save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma

C1 = lambda alpha : Gamma(1 + alpha)**(1/(1+alpha))/(1 + alpha)
C2 = lambda alpha : np.sqrt(C1(alpha)/(3 + 2*alpha))

p = lambda x,alpha : 1 - gamma.cdf(x , 1 + alpha, scale = (1+alpha)**(-1))
q = lambda x,t,alpha : p(x*C1(alpha)*t**(-1/(1+alpha)),alpha)
#q = lambda x,t,alpha : (2*x < 1)*(1 - 2*x)**(.5*C1(alpha)*t**(-1/(1+alpha)) - .5)


f1 = lambda x,t,alpha : (1 - q(x,t,alpha))**((C1(alpha) + C2(alpha)**2/2*np.log(1 - q(x,t,alpha)))*t**(-1/(1+alpha))-1)

f2 = lambda x,t,alpha : (C1(alpha) + C2(alpha)**2 * np.log(1 - q(x,t,alpha)))*q(x,t,alpha)*t**(-1/(1+alpha)) - q(x,t,alpha) + 1

f3 = lambda x,t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha))) - C2(alpha)*t**(-1/(2+2*alpha))*np.log(1- q(x,t,alpha))

h = lambda t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))

f4 = lambda x,t,alpha : 1 - norm.cdf(f3(x,t,alpha))

f5 = lambda x,t,alpha : 1/np.sqrt(2*np.pi)*q(x,t,alpha)*C2(alpha) *t**(-1/(2 + 2*alpha))*np.exp(-f3(x,t,alpha)**2/2)

f = lambda x,t,alpha :  1 - norm.cdf(h(t,alpha)) - f1(x,t,alpha)*( f2(x,t,alpha)*f4(x,t,alpha) + f5(x,t,alpha))
analytic = lambda x,t,alpha : f(x,t,alpha)

for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 150)
    # SC
    print("Computing probas for ML, parameters : alpha = %.3f"%(alpha) )
    save_path_i_ML = save_path_SC + 'alpha_%.3f'%(alpha)
    sample_sizes_ML, sample_times_ML = concatenate_sim(save_path_i_ML)
    n_samples,n_clusters = sample_times_ML.shape

    time_range = np.linspace(0,3*C1(alpha)/((k-1)**(1+alpha)),100)
    probies = probs(sample_sizes_ML,sample_times_ML,time_range,k,x)
    plt.plot(time_range,probies, label = r"SC")

    # CAML
    print("Computing probas for CAML, parameters : alpha = %.3f"%(alpha) )
    save_path_i_CAML = save_path_CASC + 'alpha_%.3f'%(alpha)
    sample_sizes_CAML, sample_times_CAML = concatenate_sim(save_path_i_CAML)
    n_samples,n_clusters = sample_times_CAML.shape
    
    probies = probs(sample_sizes_CAML,sample_times_CAML,time_range,k,x)
    plt.plot(time_range,probies, label = r"CASC")
    
    # Gaussian fluct
    print("Computing probas for CAML, parameters : alpha = %.3f"%(alpha) )
    
    ts = np.linspace(0.001,3*C1(alpha)/((k-1)**(1+alpha)),100)
    plt.plot(ts, analytic(x,ts,alpha), label = r"Analytic")
    
    plt.legend()
    plt.savefig(fig_path+parameters_file_name+'_'+str(i)+'_fig.png')