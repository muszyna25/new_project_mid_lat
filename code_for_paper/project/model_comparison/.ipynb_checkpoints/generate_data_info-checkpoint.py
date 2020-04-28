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
        #Z=hf['Z'][:]
        A=hf['A'][:]
        #print(' done',X.shape, Y.shape, Z.shape, A.shape)
        print(' done', A.shape)
        print('[+] done...!')

    #return X, Y, Z, A
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

################## MAIN ##################

f_name = 'test_files_list.txt'
lfs = get_train_val_names(f_name)
print(lfs)

path_labels = 'classifier_predict/'
data = read_data_from_csv(path_labels + 'Model_D.csv') 

#True, Prediction
Y_pred = data[:,1]
print(Y_pred.shape)

path_files = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/data_classification_regression/data_for_paper/all_data_v2/reg_source_d/' 
n = 0

for f in lfs:
    print(f)
    isExist = os.path.exists(path_files + f)
    print(isExist)

    A = read_data_hdf5(path_files + f)
    print('A', A.shape, A[0:3])

    k = A.shape[0]

    print('before n k', n, k)

    if n == 0:
        k_labels = Y_pred[n:k]
        n = k
    else:
        k_labels = Y_pred[n:n + k]
        n += k

    print('after n k', n, k)

    #print(k_labels)
    pos_samples_info = [j for i,j in zip(k_labels, A) if int(i)==1]
    #pos_samples_info = [j for i,j in zip(k_labels, A)]
    #print(pos_samples_info)
    
    with open('positive_class_info.csv', 'ab') as f:
    #with open('_positive_class_info.csv', 'ab') as f:
        np.savetxt(f,pos_samples_info, newline='\n', fmt='%s,%s,%s')




