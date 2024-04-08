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
