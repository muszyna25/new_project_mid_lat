# System imports

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

# External imports
import numpy as np
import tensorflow.python.keras
import matplotlib.pyplot as plt
import glob
import random
import csv
import sys
from random import shuffle
from tensorflow.keras.models import load_model
import pandas as pd
from scipy.stats import wilcoxon
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
import h5py
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas


#####################################
def read_data_hdf5(fn):

    with h5py.File(fn, 'r') as hf:
        keys = list(hf.keys()) 
        print(hf['X'],keys)
        X=hf['X'][:]
        Y=hf['Y'][:]
        Z=hf['Z'][:]
        A=hf['A'][:]
        print('HDF5 original file...',X.shape, Y.shape, Z.shape, A.shape)
        #print(' done', A.shape)
        print('[+] done...!')

    return X, Y, Z, A
    #return A

#####################################
def get_train_val_names(fname):

    lfs = []
    with open(fname, 'r') as f:
        lfs = f.read().splitlines()

    return lfs

#####################################
def read_data_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=',')
        #next(input_file, None)
        my_list = []
        for row in input_file:
            #my_list.append([eval(r) for r in row])
            my_list.append(row)
    
    #return np.array(my_list)
    return my_list

######################################################
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

#######################
def normalize_regres_data(Z, Y, l_minmax):

    l_consts = []
    X_n = np.empty((Z.shape[0], 3))
    #X_n = np.empty((0, 3))
    for j in range(0,3):
        tmp = l_min_max[j]
        c_min = tmp[0]
        c_max = tmp[1]
        X = Z[:,j]#.reshape(-1,1)
        _X = X
        #print("_X: ", _X)

        if c_min is None and c_max is None:
            N = (2.0 * (_X - min(_X))/(max(_X) - min(_X))) - 1.0
            l_consts.append([min(_X), max(_X)])
        else:
            N = (2.0 * (_X - c_min)/(c_max - c_min)) - 1.0

        X_n[:,j] = N

    return np.array(X_n), l_consts

################## MAIN ##################

f_name = 'file_list.txt'
lfs = get_train_val_names(f_name)
print(lfs[:10])

path_csv_files = '../regres_indices/'
path_hdf5_files = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data_v2/reg_source_d/' 
out_path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data_v2/out_regres_data'

l_min_max = [[0,117], [0,59], [0,61]]

for fn in lfs[400:]:
#for fn in lfs[401:403]:
    print(fn)
    data = read_data_from_csv(path_csv_files + fn[:-3] + '.csv') 
    print(data[0:3])

    indices = [int(i[0]) for i in data]
    print(indices[0:10])
    print('indices: ', len(indices))

    print('reading file....')
    X, Y, Z, A = read_data_hdf5(path_hdf5_files + fn)
    _X = X[indices] 
    _Y = Y[indices]
    _Z = Z[indices]
    _A = A[indices]

    #arr = f['Log list'][:]  # extract to numpy array
    #res = np.delete(arr, 1)  # delete element with index 1, i.e. second element
    #f.__delitem__('Log list')  # delete existing dataset
    #f['Log list'] = res  # reassign to dataset
    
    X_ro, Y_ro, Z_ro, A_ro = remove_outliers(_X, _Y, _Z, _A)
    Z_n, params = normalize_regres_data(Z_ro, Y_ro, l_min_max)
    print('saving file....')
    save_to_hdf5_format(X_ro, Y_ro, Z_n, A_ro, fn, out_path)
    #save_to_hdf5_format(X[indices], Y[indices], Z[indices], A[indices], fn, out_path)
    #print('HDF5 original file...',X.shape, Y.shape, Z.shape, A.shape)








