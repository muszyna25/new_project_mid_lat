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
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

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

#####################################
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
    
#######################
def normalize_regres_data(Z, Y, c_min=None, c_max=None):

    Y_POS = Y == 1

    Z1 = Z[:,0].reshape(-1,1)
    Z_d = Z1[Y_POS]
    print(Z.shape, Z1.shape, Z_d.shape)

    l_consts = []
    X_n = np.empty((Z_d.shape[0], 3)) 
    for j in range(0,3):
        X = Z[:,j].reshape(-1,1)
        _X = X[Y_POS]
        #print("_X: ", _X)

        if c_min is None and c_max is None:
            N = (2.0 * (_X - min(_X))/(max(_X) - min(_X))) - 1.0 
            l_consts.append([min(_X), max(_X)])
        else:
            N = (2.0 * (_X - c_min)/(c_max - c_min)) - 1.0 

        X_n[:,j] = N

    return np.array(X_n), l_consts 

#######################
def normalize_data(X):

    X_n = np.empty((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))

    for i in range(0,X.shape[3]):
        for j in range(0,X.shape[0]):
            X_n[j,:,:,i]=normalize(X[j,:,:,i], norm='l2')

    return X_n

#######################
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

def get_k_folds(X):
    kfolds = 10
    l_k_folds = []
    N = np.arange(X.shape[0])
    for i in range(0,kfolds):
        select_folds = np.split(N, [int(.8 * len(N)), int(.9 * len(N))])
        l_k_folds.append(select_folds)
        N = np.roll(N, int(N.shape[0]*0.10))
    return l_k_folds


###################### MAIN #####################################

if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_fname = sys.argv[3]
    task_type = int(sys.argv[4])

    print(sys.argv[0], input_path, output_path, input_fname, task_type)
    
    X, Y, Z, A = read_data_hdf5(input_path + '/' + input_fname)
    
    #Remove outliers based on Z variable.
    if task_type == True: 
        print('TASK0', task_type)
        X, Y, Z, A = remove_outliers(X, Y, Z, A)

    X = normalize_data(X)
   
    ### Divide data into k folds ###
    l_k_folds = get_k_folds(X)

    ### Normalize data: each image individually and x,y,r feature-wise ###
    ### Split data into chunks for keras data generator (streaming) ###
    msubsets = 10 
    lsets = ['train','val','test']
    #order: train, val, test
    for i in range(0,len(l_k_folds)): #round
        xf = l_k_folds[i]
        lconsts = []
        for j in range(0,len(xf)): #train/val/test
            s = xf[j]  
            print('Sets', s.shape)

            _X = X[s,]; _Y = Y[s,]; _Z = Z[s,]; _A = A[s,]

            if task_type == True:
                print('TASK1', task_type)
                if j == 0: #Normalize training data and get consts to use them in val and train normalization.
                    _Z, lconsts = normalize_regres_data(_Z, _Y)
                else:                                               #c_min          c_max
                    _Z = normalize_regres_data(_Z, _Y, lconsts[j][0], lconsts[j][1])[0] #Just return feature vectors

            N_new = np.arange(_X.shape[0])
            chunks = np.array_split(N_new, msubsets)
            for k in range(0, len(chunks)): #chunks
                df = chunks[k]
                print('shape', df.shape)
                fname = str(i) + '_' + lsets[j] + '_' + str(k) + '_' + '.h5'
                print('check all:', _X.shape, _Y.shape, _Z.shape, _A.shape)
                save_to_hdf5_format(_X[df,:,:,:], _Y[df,], _Z[df,], _A[df,], fname, output_path)

    print(sys.argv[0], "[+++] DONE!")



















    '''
    ### Split data into chunks for keras data generator (streaming) ###
    msubsets = 10 
    lsets = ['train','val','test']
    #order: train, val, test
    for i in range(0,len(l_k_folds)): #round
        xf = l_k_folds[i]
        for j in range(0,len(xf)): #train/val/test
            s = xf[j]                                   #Or do normalization here!!!!
            print('Sets', s.shape)
            chunks = np.array_split(s, msubsets)
            print('Chunks', chunks)
            for k in range(0, len(chunks)): #chunks
                df = chunks[k]
                print('shape', df.shape)
                fname = str(i) + '_' + lsets[j] + '_' + str(k) + '_' + '.h5'
                save_to_hdf5_format(X[df,:,:,:], Y[df,], Z[df,], A[df,], fname, output_path)
    '''



