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

def read_data_hdf5(fn):
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(keys)
    X = hf['X'].value
    Y = hf['Y'].value
    return X, Y

### Split data into train/val/test
def split_dataset(n):
    print('Split data indices: ', n)
    X = np.arange(n)
    Y = np.arange(n)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=111)
    X_train_, X_val, Y_train_, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=123)
    
    print('[+] done...!')

    return X_train_, X_val, X_test, Y_train_, Y_val, Y_test

### Save to hdf
def save_to_hdf5_format(X, Y, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape)

    S = X
    L = Y

    print('Selected data shapes: ', S.shape, L.shape)

    fn = out_path + '/' + fname

    if os.path.exists(fn) == False:
        print('Create %s' %fn)
        hf = h5py.File(fn, 'w')
        hf.create_dataset('X', data=S, maxshape=(None, S.shape[1], S.shape[2], S.shape[3]))
        hf.create_dataset('Y', data=L, maxshape=(None, 1))
        hf.close()
    elif os.path.exists(fn) == True:
        print('Update %s' %fn)
        hf = h5py.File(fn, 'a')
        hf['X'].resize((hf['X'].shape[0] + S.shape[0]), axis = 0)
        hf['X'][-S.shape[0]:] = S
        hf['Y'].resize((hf['Y'].shape[0] + L.shape[0]), axis = 0)
        hf['Y'][-L.shape[0]:] = L
        hf.close()
    print('[+] done...!')

#### MAIN ####

if __name__ == "__main__":

    fn_list = sys.argv[1]
    dir_path_bin_masks = sys.argv[2]
    dir_path_era_raw = sys.argv[3]

    df_list = read_file_list(fn_list)
    
    #out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/train_val_test'
    out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/extracted_raw_data_train_val_test/t'
    
    fn_train = 'train.h5'
    fn_val = 'val.h5'
    fn_test = 'test.h5'

    n_train = 168
    n_val = 24
    n_test = 24

    #TRAIN SET
    for i in range(0, n_train):
        f = df_list[i]
        print(f)
        X_bm, Y_bm = read_data_hdf5(dir_path_bin_masks + '/' + f)
        X_raw, Y_raw = read_data_hdf5(dir_path_era_raw + '/' + f)

        Y = np.reshape(Y_bm, (Y_bm.shape[0],1))
        X_tmp = np.swapaxes(X_raw,1,2)
        X = np.swapaxes(X_tmp,2,3)

        save_to_hdf5_format(X, Y, fn_train, out_path)

    #VAL SET
    for i in range(n_train, n_train + n_val):
        f = df_list[i]
        print(f)
        X_bm, Y_bm = read_data_hdf5(dir_path_bin_masks + '/' + f)
        X_raw, Y_raw = read_data_hdf5(dir_path_era_raw + '/' + f)

        Y = np.reshape(Y_bm, (Y_bm.shape[0],1))
        X_tmp = np.swapaxes(X_raw,1,2)
        X = np.swapaxes(X_tmp,2,3)

        save_to_hdf5_format(X, Y, fn_val, out_path)

    #TEST SET
    for i in range(n_train + n_test, len(df_list)):
        f = df_list[i]
        print(f)
        X_bm, Y_bm = read_data_hdf5(dir_path_bin_masks + '/' + f)
        X_raw, Y_raw = read_data_hdf5(dir_path_era_raw + '/' + f)

        Y = np.reshape(Y_bm, (Y_bm.shape[0],1))
        X_tmp = np.swapaxes(X_raw,1,2)
        X = np.swapaxes(X_tmp,2,3)

        save_to_hdf5_format(X, Y, fn_test, out_path)





