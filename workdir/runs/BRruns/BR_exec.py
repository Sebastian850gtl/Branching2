import os,sys,subprocess
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
from Brownian import Modelv3 as Model


parameters = sys.argv[1] # parameters file
number_of_runs = int(sys.argv[2])  # Simulation tag

#Files locations

save_path = '../../results/'+file_name+'/'
fig_path = '../../results/fig/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#Import parameters

param_module = importlib.import_module(parameters)

samples_by_run = param_module.n_samples//number_of_runs

print(samples_by_run)
for run_tag in range(number_of_runs):
    #command = "nohup python3 workdir/runs/BRruns/BR_core.py "+ str(parameters) +" "+str(run_tag)+" "+str(samples_by_run) + "&>out.txt &"
    subprocess.Popen(["nohup","python3","workdir/runs/BRruns/BR_core.py",parameters,str(run_tag),str(samples_by_run)],
                    stdout=open('/dev/null', 'w'),
                    stderr=open('logfile.log', 'a'),
                    preexec_fn=os.setpgrp)

