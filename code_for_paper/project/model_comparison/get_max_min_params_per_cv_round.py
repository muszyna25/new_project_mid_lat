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

for fn, i in zip(lfs[0:201], range(0,201)):
    print(fn)
    data = read_data_from_csv(path_csv_files + fn[:-3] + '.csv') 
    print(data[0:3])







