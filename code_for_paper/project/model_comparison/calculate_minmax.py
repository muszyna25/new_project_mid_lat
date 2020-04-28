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
        #X=hf['X'][:]
        #Y=hf['Y'][:]
        Z=hf['Z'][:]
        #A=hf['A'][:]
        #print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print(' done', Z.shape)
        print('[+] done...!')

    #return X, Y, Z, A
    return Z

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

################## MAIN ##################

f_name = 'file_list.txt'
lfs = get_train_val_names(f_name)
print(lfs[:10])

path_csv_files = 'regres_indices/'
path_hdf5_files = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data_v2/reg_source_d/' 

#for fn in lfs[0:2]:
l_minmax_0 = []
l_minmax_1 = []
l_minmax_2 = []
for fn, i in zip(lfs[0:201], range(0,201)):
    print(fn)
    data = read_data_from_csv(path_csv_files + fn[:-3] + '.csv') 
    print(data[0:3])

    indices = [int(i[0]) for i in data]
    print(indices[0:10])
    print('indices: ', len(indices))

    print('reading file....')
    Z = read_data_hdf5(path_hdf5_files + fn)
    Z_pos = Z[indices]
    print(Z_pos.shape)

    if i % 20 == 0 and i != 0:
        print('Shapes:', len(l_minmax_0), len(l_minmax_1), len(l_minmax_2))
        #c_min = min() 
        #c_max =

        with open('min_max_params/' + str(i) + '_' + str(0) + '_min_max_params_for_regres.csv', 'ab') as fd:
             np.savetxt(fd, l_minmax_0, fmt='%s,%s', delimiter=',')
        with open('min_max_params/' + str(i) + '_' + str(1) + '_min_max_params_for_regres.csv', 'ab') as fd:
             np.savetxt(fd, l_minmax_1, fmt='%s,%s', delimiter=',')
        with open('min_max_params/' + str(i) + '_' + str(2) + '_min_max_params_for_regres.csv', 'ab') as fd:
             np.savetxt(fd, l_minmax_2, fmt='%s,%s', delimiter=',')
        l_minmax_0 = []
        l_minmax_1 = []
        l_minmax_2 = []

    Z_n_0 = Z[:,0].reshape(-1,1)
    l_minmax_0.append([int(min(Z_n_0)), int(max(Z_n_0))])

    Z_n_1 = Z[:,1].reshape(-1,1)
    l_minmax_1.append([int(min(Z_n_1)), int(max(Z_n_1))])

    Z_n_2 = Z[:,2].reshape(-1,1)
    l_minmax_2.append([int(min(Z_n_2)), int(max(Z_n_2))])


    #    if i % 20 == 0 and i != 0:
    #        with open('min_max_params/' + str(i) + '_' + str(j) + '_min_max_params_for_regres.csv', 'ab') as fd:
    #            np.savetxt(fd, l_minmax, fmt='%s,%s', delimiter=',')




