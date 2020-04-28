
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
from statistics import mean

def read_data_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=',')
        #next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)

    return np.array(my_list)

##########################
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
        X=h5f['X'][:]
        Y=h5f['Y'][:]
        print(' done',X.shape, Y.shape)

    return X, Y

###### MAIN ##########

m_type0 = 'D'
m_type1 = 'E'
i='SP'
nv = 2

f_name0 = 'loc_reg_model_' + m_type0 + '.h5'
fd0 = h5py.File(f_name0, 'r')
print(fd0)

f_name1 = 'loc_reg_model_' + m_type1 + '.h5'
fd1 = h5py.File(f_name1, 'r')
print(fd1)

#for i in fd0.keys():
print('Name:', i)

y_t = fd0[i]['Y_t']
y_p0 = fd0[i]['Y_p']
y_p1 = fd1[i]['Y_p']

print(y_t.shape)
print(y_p0.shape)
print(y_p1.shape)

###### Statistical Hypothesis Testing #####
stat, p = wilcoxon(y_p0[:,nv], y_p1[:,nv], correction=False)
#stat, p = wilcoxon(y_pred_0, y_pred_1, correction=True)

print('Statistics=%.3f, p=%.12f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

fd0.close()
fd1.close()


