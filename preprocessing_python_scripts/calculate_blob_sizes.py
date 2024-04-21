#!/usr/bin/env python

import sys 
import os
from netCDF4 import Dataset
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt

#####################################
def read_netcdf_file(fname, varname): #Variables names: e.g., 'lon', 'lat', 'prw'
    print('read_netcdf_file', fname)
    fh = Dataset(fname, mode='r')
    var_netcdf = fh.variables[varname][:] #Retrieves a given variable by name.
    fh.close()
    return var_netcdf

#####################################
def read_file_list(name): 
    fn_list = []
    with open(name, 'r') as filehandle:
        for line in filehandle:
            fn = line[:-1]
            fn_list.append(fn)
    return fn_list

################################################################################################
#   INPUT:
#       fn_list -- text file with list of ETHZ nc files 
#       dir_path -- path location of ETHZ nc datafiles
#
#   OUTPUT:
#       output -- text file with bloc sizes 
################################################################################################

if __name__ == "__main__":

    fn_list = sys.argv[1]
    dir_path = sys.argv[2]
    df_list = read_file_list(fn_list)

    file_info = []
    
    debug = False
    l_blob_sizes = []
    
    for i in range(0, len(df_list)):
        f = df_list[i]
        print(f)

        #Read time variable
        time_var = read_netcdf_file(dir_path + f, 'time')
        print(time_var.shape[0])

        #Read flag variable (binary mask)
        flag_var = read_netcdf_file(dir_path + f, 'FLAG')
        print(flag_var.shape)

        #Latitudes cut-offs
        north_lat_cut_off = 15
        south_lat_cut_off = 165

        for j in range(0, time_var.shape[0]):
            #Select a timestep
            img = flag_var[j][north_lat_cut_off:south_lat_cut_off][:]

            blobs, n_blobs = measure.label(img, neighbors=4, background=0, return_num=True)
            print('#blobs ', n_blobs)
            
            d_blobs = dict(zip(range(1,n_blobs+1), np.zeros(n_blobs, int)))

            for x in range(0, blobs.shape[0]):
                for y in range(0, blobs.shape[1]):
                    val = blobs[x][y]
                    if val != 0:
                        d_blobs[val] += 1
                    
            print(d_blobs)
        
            for k in d_blobs.keys():
                l_blob_sizes.append(d_blobs[k])

            if debug:
                #Show a binary image and connected components
                plt.figure()
                plt.subplot(121)
                plt.imshow(img)
                plt.subplot(122)
                plt.imshow(blobs, cmap='nipy_spectral')
                plt.show()

            output_fn = 'blob_sizes.txt'
            #Save list to text file.
            with open(output_fn, 'w') as filehandle:
                for listitem in l_blob_sizes:
                    filehandle.write('%s\n' % listitem)
                    
        print('Blob sizes: ', len(l_blob_sizes))

