import os
import numpy as np

def concatenate_sim(results_path):
    tmp_results_path = results_path+'/tmp/'
    if os.path.exists(tmp_results_path):
        print("Concatenating results in file "+tmp_results_path)
        directory = os.fsencode(tmp_results_path)
        list_of_directories = os.listdir(directory)
        print("Number of temporary instances : %s"%(len(list_of_directories)))
        
        first_file = os.fsdecode(os.listdir(directory)[0])
        #print("Treating file :"+first_file)
        first_array = np.load(tmp_results_path+first_file)

        if len(first_array.shape) == 3:
            n_samples,n_clusters,_ = first_array.shape
            sample_sizes = first_array
            sample_times = np.empty([0,n_clusters])
        else:
            n_samples,n_clusters = first_array.shape
            sample_times = first_array
            sample_sizes = np.empty([0,n_clusters,n_clusters])
        #print(" Number of samples %s,"%(n_samples))
        print(" Number of initial particles : %s"%(n_clusters))
        
        for file in list_of_directories[1:]:
            filename = os.fsdecode(file)
            #print("Treating file :"+filename)
            array_i = np.load(tmp_results_path+filename)
            if len(array_i.shape) == 3:
                sample_sizes = np.concatenate((sample_sizes,array_i),axis = 0)
                #n_samples,_,_ = sample_sizes.shape
                #print(" Number of samples %s,"%(n_samples))
            else:
                sample_times = np.concatenate((sample_times,array_i),axis = 0)
                #n_samples,_,_ = sample_sizes.shape
                #print(" Number fo samples %s,"%(n_samples))
        n_samples,_ = sample_times.shape
        print(" Number of final samples %s,"%(n_samples))
        np.save(results_path +'/_sizes.npy',sample_sizes)
        np.save(results_path +'/_times.npy',sample_times)
        print("Deleting temporary file")
        for file in list_of_directories:
            filename = os.fsdecode(file)
            os.remove(tmp_results_path+filename)
        os.rmdir(tmp_results_path)
        return sample_sizes, sample_times
    else:
        print("No temporary files")
        print("Loading results")
        try:
            sample_sizes = np.load(results_path +'/_sizes.npy')
            sample_times = np.load(results_path +'/_times.npy')
            n_samples,_ = sample_times.shape
            print(" Number of final samples %s,"%(n_samples))
            return sample_sizes, sample_times
        except:
            raise ValueError("No sample_files")


