#!/usr/bin/env python

import sys 
import os
from netCDF4 import Dataset
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt
import operator
import h5py

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

def swap_halves(field,index=2):
    print(field.shape)
    rank = len(field.shape)
    # define a variable to access all values of an array
    accessor = rank*[slice(None,None,None)]

    # define the slices for the 1st and 2nd halves
    nhalf = int(field.shape[index]/2)
    first_half_slice = slice(0,nhalf+1,None) # nhalf + 1, because it longitude is not even number
    second_half_slice = slice(nhalf,None,None)

    # define the N-dimensional accessors for the 1st half slice
    first_half_accessor = list(accessor)
    first_half_accessor[index] = first_half_slice
    # convert to tuple so it can be used as a numpy index
    first_half_accessor = tuple(first_half_accessor)

    # define the N-dimensional accessors for the 2nd half slice
    second_half_accessor = list(accessor)
    second_half_accessor[index] = second_half_slice
    # convert to tuple so it can be used as a numpy index
    second_half_accessor = tuple(second_half_accessor)

    # copy the array
    output_array = np.array(field)

    # swap halves
    output_array[first_half_accessor] = field[second_half_accessor]
    output_array[second_half_accessor] = field[first_half_accessor]

    return output_array

### Shift data in netCDF file
def shift_binary_mask(x_fn, var_name='longitude'):
    offset = 180.0
    print('[+] File: %s' %x_fn)
    fh = Dataset(x_fn, mode='r+')
    with fh as fout:
        lon = np.array([])
        lon = fh.variables[var_name][:]
        for i in range(0, len(lon)): # Shift coordinates by offset, i.e. 180
            lon[i] = lon[i] + offset

        flag = fh.variables['FLAG'][:]
        flag_t = swap_halves(flag) # Swap the east hemisphere with the west one.

        fout.variables[var_name][:] = lon
        fout.variables['FLAG'][:] = flag_t
    print('[+] done...!')

##### MAIN #####
if __name__ == "__main__":

    fn_list = sys.argv[1]
    dir_path = sys.argv[2]
    df_list = read_file_list(fn_list)

    for i in range(0, len(df_list)):
        f = df_list[i]
        print(f)

        #Read flag variable (binary mask)
        shift_binary_mask(dir_path + '/' + f)
        



