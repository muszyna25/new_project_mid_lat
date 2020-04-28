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

def read_results_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=',')
        next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)
    
    return my_list

##########################
#def get_train_val_names_v2(fname, start_ind, end_ind, n_files):
def get_train_val_names_v2(fname, start_ind, n_files):
    lfs = []
    with open(fname, 'r') as f:
        #lfs = f.readlines() 
        lfs = f.read().splitlines()

    #random.seed(4)
    l_train = lfs[start_ind - 1:start_ind - 1 + n_files]
    #l_val = lfs[end_ind - 1:end_ind - 1 + n_files]
    #print('Original order ', l_train, l_val)
    #shuffle(l_train)
    #shuffle(l_val)

    #return l_train, l_val
    return l_train

#########################
def read_hdf5_file(fname):

    with h5py.File(fname, 'r') as h5f:
        A=h5f['A'][:]
        #Y=h5f['Y'][:]
        #print(' done',X.shape, Y.shape)
        print(' done',A.shape)
 
    return A 

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
            my_list.append(row)

    return np.array(my_list)

#####################################
def save_to_hdf5_format(D, fname, out_path):

    #fn = out_path + '/' + fname + '.h5'
    fn = fname + '.h5'
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    # keys = list(D.keys()) # 'NA', etc.

    for dk in D.keys():
        g = hf.create_group(dk)
        for k in D[dk].keys():
            _data = D[dk][k]
            print(dk, k)
            g.create_dataset(k, data=_data)

    hf.close()
    print('[+] done...!')

###### MAIN ##########

f_name = 'test_files_list.txt'
lfs = get_train_val_names(f_name)
print(lfs)

path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/'
input_dir = path + 'data_classification_regression/data_for_paper/all_data_v2/source_d/'

out_path = ''

REG_DATA = {'NC': {'Y_t': [], 'Y_p': []}, 'NP': {'Y_t': [], 'Y_p': []}, 'NA': {'Y_t': [], 'Y_p': []}, 
        'SI': {'Y_t': [], 'Y_p': []}, 'SP': {'Y_t': [], 'Y_p': []}, 'SA': {'Y_t': [], 'Y_p': []}}

TIME_DATA = {str(k):{'Y_t': [], 'Y_p': []} for k in range(1980,2017) if k!= 1999} 

print(TIME_DATA)
print(REG_DATA)

path_labels = '../classifier_predict/'

m_type = 'E'
data = read_data_from_csv(path_labels + 'Model_' + m_type + '.csv') # Check this line for different models

Y_t = data[:,0]
Y_p = data[:,1]
counter = 0

for i in lfs:
    #for i in lfs[0:4]:
    print('Name:', i)

    A = read_hdf5_file(input_dir + i)
    print(A.shape)
    print(A[0:10])

    for a in A:
        _y = a[0].decode('utf-8')
        year = _y[6:-3]
        reg = a[2].decode('utf-8')
        #print(year, reg)

        REG_DATA[reg]['Y_t'].append(int(Y_t[counter]))
        REG_DATA[reg]['Y_p'].append(int(Y_p[counter]))

        TIME_DATA[year]['Y_t'].append(int(Y_t[counter]))
        TIME_DATA[year]['Y_p'].append(int(Y_p[counter]))
        counter += 1

    print(TIME_DATA['1994'])
    print(REG_DATA['NC'])

save_to_hdf5_format(REG_DATA, 'reg_model_' + m_type, out_path)
save_to_hdf5_format(TIME_DATA, 'time_model_' + m_type, out_path)










