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
from ML import Coalescence
from CAML import CAML

#Files locations


fig_path = '../../results/fig/BR_beta/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

parameters_file_name = sys.argv[1]

#Parameters

param_module = importlib.import_module(parameters_file_name)

radius = param_module.radius
alpha_range = param_module.alpha_range
x = param_module.x
k = param_module.k

BR_folder = param_module.BR_folder
SC_folder = param_module.SC_folder
CASC_folder = param_module.CASC_folder

save_path_BR = '../../results/' + BR_folder + '/'
save_path_SC = '../../results/' + SC_folder + '/'
save_path_CASC = '../../results/' + CASC_folder + '/'

from compute_probas import probs
from concatenator import concatenate_sim
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm, gamma
from scipy.special import gamma as Gamma

C1 = lambda alpha : Gamma(1 + alpha)**(1/(1+alpha))/(1 + alpha)
C2 = lambda alpha : np.sqrt(C1(alpha)/(3 + 2*alpha))

p = lambda x,alpha : 1 - gamma.cdf(x , 1 + alpha, scale = (1+alpha)**(-1))
q = lambda x,t,alpha : p(x*(C1(alpha)*t**(-1/(1+alpha))),alpha)
f1 = lambda x,t,alpha : (1 - q(x,t,alpha))**((C1(alpha) + C2(alpha)**2/2*np.log(1 - q(x,t,alpha)))*t**(-1/(1+alpha))-1)

f2 = lambda x,t,alpha : (C1(alpha) + C2(alpha)**2 * np.log(1 - q(x,t,alpha)))*q(x,t,alpha)*t**(-1/(1+alpha)) - q(x,t,alpha) + 1

f3 = lambda x,t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha))) - C2(alpha)*t**(-1/(2+2*alpha))*np.log(1- q(x,t,alpha))

h = lambda t,alpha : (k-1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))

f4 = lambda x,t,alpha : 1 - norm.cdf(f3(x,t,alpha))

f5 = lambda x,t,alpha : 1/np.sqrt(2*np.pi)*q(x,t,alpha)*C2(alpha) *t**(-1/(2 + 2*alpha))*np.exp(-f3(x,t,alpha)**2/2)

f = lambda x,t,alpha :  1 - norm.cdf(h(t,alpha)) - f1(x,t,alpha)*( f2(x,t,alpha)*f4(x,t,alpha) + f5(x,t,alpha))
analytic = lambda x,t,alpha : f(x,t,alpha)

#analytic2 = lambda x,t,alpha : (q(x,t,alpha))*(1 - norm.cdf((1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))))

analytic2 = lambda x,t,alpha : (q(x,t,alpha))**2 *(1 - norm.cdf((1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))))

# analytic3 = lambda x,t,alpha : 1 / Gamma(1 + alpha)**2* (1 + alpha)**(2*alpha)* x**(2*alpha)*(1 - 2*x)**((1 + alpha)*C1(alpha)*t**(-1/(1+alpha))-1-2*alpha)* (1 - norm.cdf((1 - C1(alpha)*t**(-1/(1+alpha)))/(C2(alpha)*t**(-1/(2+2*alpha)))))

analytic3 = lambda x,t,alpha : (1 - 2*x)**(1/t) * np.exp(-t)
print("Hello")

colors = ["crimson","darkgreen","darkcyan"]
markers = ["^","*","h"]
linestyles = ["dashed","dotted","dashdot"]

beta_range = [0,1/2,1]
for i,alpha in enumerate(alpha_range):
    plt.figure(dpi = 200)
    #plt.title(r"$x = %.3f$ and $\alpha = %.3f$"%(x,alpha))
    plt.xlabel("Time")
    plt.ylabel("Branching probability")
    for j,beta in enumerate(beta_range):

        color, linestyle, marker = colors[j], linestyles[j], markers[j]

        print("Computing probas for BR, parameters : alpha = %.3f, beta = %.3f"%(alpha,beta))
        save_path_i_BR = save_path_BR + "alpha_beta_%.3f_%.3f"%(alpha,beta)
        sample_sizes_BR, sample_times_BR = concatenate_sim(save_path_i_BR)

        n_samples, n_clusters = sample_times_BR.shape
        time_range_BR = np.linspace(0,3*np.mean(sample_times_BR[:,-1]),300)

        probies = probs(sample_sizes_BR,sample_times_BR,time_range_BR,k,x)
        #print(probies)
        time_range =  time_range_BR *(n_clusters)**(-alpha) *1/(-np.log(2*radius) - beta * np.log(n_clusters/2) + np.log(2)) #To be able to compare BR with SC and CASC we divide be the costant in front of the kernel.
        plt.plot(time_range,probies, label = r"\beta = %.2f"%(beta), color = color,marker = marker, linestyle = linestyle ,markevery=30)


    plt.legend()
    plt.savefig(fig_path+parameters_file_name+'_%.3f_fig_BRbeta.png'%(alpha))