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
from sklearn.model_selection import train_test_split

#####################################
def read_file_list(name): 
    fn_list = []
    with open(name, 'r') as filehandle:
        for line in filehandle:
            fn = line[:-1]
            fn_list.append(fn)
    return fn_list

#####################################
def read_data_hdf5(fn):
    hf = h5py.File(fn, 'r')
    keys = list(hf.keys()) # 'NA', etc.
    print(keys)
    X = hf['X'].value
    Y = hf['Y'].value
    return X, Y

#####################################
### Save to hdf
def save_to_hdf5_format(X, Y, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape)
    S = X
    L = Y
    print('Selected data shapes: ', S.shape, L.shape)
    fn = out_path + '/' + fname
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('X', data=S)
    hf.create_dataset('Y', data=L)
    hf.close()
    print('[+] done...!')

#####################################
### Split data into train/val/test
def split_data_indices(n):
    print('Split data indices: ', n)
    X = np.arange(n)
    Y = np.arange(n)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=111)
    X_train_, X_val, Y_train_, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=123)
    
    print('[+] done...!')

    return X_train_, X_val, X_test, Y_train_, Y_val, Y_test    
    
###################### MAIN #####################################

if __name__ == "__main__":

    input_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    output_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/corrected_big_train_val_test/pv'
    
    f_name = 'second_shuffle_after_resampled_pv_data.h5'
    
    X, Y = read_data_hdf5(input_path + '/' + f_name)

    fnames = ['train.h5', 'val.h5', 'test.h5']

    X_train_, X_val, X_test, Y_train_, Y_val, Y_test = split_data_indices(X.shape[0])
    
    save_to_hdf5_format(X[X_train_,:,:,:], Y[X_train_,], fnames[0], output_path)
    save_to_hdf5_format(X[X_val,:,:,:], Y[X_val,], fnames[1], output_path)
    save_to_hdf5_format(X[X_test,:,:,:], Y[X_test,], fnames[2], output_path)
    





