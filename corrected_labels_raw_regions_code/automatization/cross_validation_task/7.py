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
from sklearn.preprocessing import normalize
import math

#####################################
def read_data_hdf5(fn):
    sys.stdout.flush()
    
    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys()) # 'NA', etc.
        print(hf['X'],keys)
        X=hf['X'][:]
        Y=hf['Y'][:]
        Z=hf['Z'][:]
        A=hf['A'][:]
        print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print('[+] done...!')
    
    return X, Y, Z, A

### Save to hdf
def save_to_hdf5_format(X, Y, Z, A, fname, out_path):

    print('Data shapes: ', X.shape, Y.shape, Z.shape, A.shape)
    S = X
    L = Y
    V = Z
    I = A
    
    print('Selected data shapes: ', S.shape, L.shape, V.shape, I.shape)
    fn = out_path + '/' + fname
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('X', data=S)
    hf.create_dataset('Y', data=L)
    hf.create_dataset('Z', data=V)
    hf.create_dataset('A', data=I)
    hf.close()
    print('[+] done...!')

#######################
def normalize_data(Z, Y):

    Y_POS = Y == 1

    Z1 = Z[:,0].reshape(-1,1)
    Z_d = Z1[Y_POS]
    print(Z.shape, Z1.shape, Z_d.shape)

    X_n = np.empty((Z_d.shape[0], 3)) 
    for j in range(0,3):
        X = Z[:,j].reshape(-1,1)
        _X = X[Y_POS]
        N = (2.0 * (_X - min(_X))/(max(_X) - min(_X))) - 1.0 
        X_n[:,j] = N
        #print('X,N', X.shape, N.shape)
        #print(np.c_[_X[800:1000],N[800:1000]])

    return np.array(X_n)

def remove_outliers(X, Y, Z, A):
    print('Shapes before: ', X.shape, Y.shape, Z.shape, A.shape)
    
    inds_0 = np.argwhere((Z[:,0] <= 5.0))
    Z = np.delete(Z, inds_0, axis=0)
    X = np.delete(X, inds_0, axis=0)
    Y = np.delete(Y, inds_0, axis=0)
    A = np.delete(A, inds_0, axis=0)
    
    inds_1 = np.argwhere((Z[:,0] >= 115.0))
    Z = np.delete(Z, inds_1, axis=0)
    X = np.delete(X, inds_1, axis=0)
    Y = np.delete(Y, inds_1, axis=0)
    A = np.delete(A, inds_1, axis=0)

    inds_2 = np.argwhere((Z[:,2] <= 10.0))
    Z = np.delete(Z, inds_2, axis=0)
    X = np.delete(X, inds_2, axis=0)
    Y = np.delete(Y, inds_2, axis=0)
    A = np.delete(A, inds_2, axis=0)

    inds_3 = np.argwhere((Z[:,2] >= 40.0))
    Z = np.delete(Z, inds_3, axis=0)
    X = np.delete(X, inds_3, axis=0)
    Y = np.delete(Y, inds_3, axis=0)
    A = np.delete(A, inds_3, axis=0)
        
    print('Z shape after: ', X.shape, Y.shape, Z.shape, A.shape)
    
    return X,Y,Z,A
    
###################### MAIN #####################################

if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    var = sys.argv[3]
    
    print(sys.argv[0], input_path, output_path, var)

    fnames = ['train.h5', 'val.h5', 'test.h5']
    
    X, Y, Z, A = read_data_hdf5(input_path + '/' + fnames[0])
    print('X, Y, Z, A', X.shape, Y.shape, Z.shape, A.shape)
    n_lvs = 8
    X,Y,Z,A = remove_outliers(X, Y, Z, A)
    Z = normalize_data(Z,Y)
    
    X_n = np.empty((X.shape[0], X.shape[1], X.shape[2], X.shape[3])) 
    
    for i in range(0,n_lvs):
        for j in range(0,X.shape[0]):
            X_n[j,:,:,i]=normalize(X[j,:,:,i], norm='l2')
    
    print('Shape check: ', X_n.shape)
    save_to_hdf5_format(X_n, Y, Z, A, var + '_norm_train.h5', output_path)
    
    ### Normalize val and train
    
    X_v, Y_v, Z_v, A_v = read_data_hdf5(input_path + '/' + fnames[1])
    print('X, Y, Z, A', X_v.shape, Y_v.shape, Z_v.shape, A_v.shape)
    X_v,Y_v,Z_v,A_v = remove_outliers(X_v, Y_v, Z_v, A_v)
    Z_v = normalize_data(Z_v,Y_v)
    
    X_val = np.empty((X_v.shape[0], X_v.shape[1], X_v.shape[2], X_v.shape[3])) 
    
    for i in range(0,n_lvs):
        for j in range(0,X_v.shape[0]):
            X_val[j,:,:,i]=normalize(X_v[j,:,:,i], norm='l2')
    
    print('Shape check: ', X_val.shape)
    save_to_hdf5_format(X_val, Y_v, Z_v, A_v, var + '_norm_val.h5', output_path)
    
    X_t, Y_t, Z_t, A_t = read_data_hdf5(input_path + '/' + fnames[2])
    print('X, Y, Z, A', X_t.shape, Y_t.shape, Z_t.shape, A_t.shape)
    X_t,Y_t,Z_t,A_t = remove_outliers(X_t, Y_t, Z_t, A_t)
    Z_t = normalize_data(Z_t,Y_t)
    
    X_test = np.empty((X_t.shape[0], X_t.shape[1], X_t.shape[2], X_t.shape[3])) 
    
    for i in range(0,n_lvs):
        for j in range(0,X_t.shape[0]):
            X_test[j,:,:,i]=normalize(X_t[j,:,:,i], norm='l2')
    
    print('Shape check: ', X_test.shape)
    save_to_hdf5_format(X_test, Y_t, Z_t, A_t, var + '_norm_test.h5', output_path)
    
    print(sys.argv[0], "[+++] DONE!")



