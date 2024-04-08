import os,sys


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')
wd = os.getcwd() # Defines the working directory

import numpy as np

file_name = sys.argv[1]  # loc of the file after workdir
results_path = '../../results/'+file_name+'/tmp'
if os.path.exists(results_path):
    directory = os.fsencode(results_path)
    list_of_directories = os.listdir(directory)
    print("Number of temporary instances : %s"%(len(list_of_directories)))
    
    first_sample_sizes_file = os.fsdecode(os.listdir(directory)[0])
    print("Treating file :"+first_sample_sizes_file)
    sample_sizes = np.load(first_sample_sizes_file)

    first_sample_times_file = os.fsdecode(os.listdir(directory)[1])
    print("Treating file :"+first_sample_times_file)
    sample_times = np.load(first_sample_times_file)
    
    n_samples, n_clusters = sample_times.shape
    print(" Number of initial particles : %s"%(n_clusters))
    print(" Number fo samples %s,"%(n_samples))
    for i,file in enumerate(os.listdir(directory),start=2):
        tmp_filename = os.fsdecode(file)
        print("Treating file :"+tmp_filename)
        
        if i%2 == 0:
            sample_sizes_i = tmp_filename
            sample_sizes = np.concatenate((sample_sizes,sample_sizes_i),axis = 0)
            n_samples,_,_ = sample_sizes.shape
            print(" Number fo samples %s,"%(n_samples))
        else:
            sample_times_i = tmp_filename
            sample_times = np.concatenate((sample_times,sample_times_i),axis = 0)
            #n_samples,_,_ = sample_sizes.shape
            #print(" Number fo samples %s,"%(n_samples))


