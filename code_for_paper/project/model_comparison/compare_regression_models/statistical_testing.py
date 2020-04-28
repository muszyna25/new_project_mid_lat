# System imports
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

# External imports
import numpy as np
import tensorflow.python.keras
import matplotlib.pyplot as plt
import math
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
        X=h5f['Y_t'][:]
        Y=h5f['Y_p'][:]
        print(' done',X.shape, Y.shape)
 
    return X, Y

def vec_mag(x): 
    return math.sqrt(sum(i**2 for i in x))

###### MAIN ##########

path = 'regressor_predict/avgs/'

#data_0 = read_data_from_csv(path + 'Model_C.csv') 
#data_1 = read_data_from_csv(path + 'Model_A.csv') 

Y_true_0, Y_pred_0 = read_hdf5_file(path + 'Model_D.h5')
Y_true_1, Y_pred_1 = read_hdf5_file(path + 'Model_E.h5')

y_pred_0 = [vec_mag(x) for x in Y_pred_0]
y_pred_1 = [vec_mag(x) for x in Y_pred_1]
print(y_pred_0[:10], y_pred_1[:10])

stat, p = wilcoxon(y_pred_0, y_pred_1)
#stat, p = wilcoxon(y_pred_0, y_pred_1, correction=True)

print('Statistics=%.3f, p=%.12f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distribution (fail to reject H0)')
else:
	print('Different distribution (reject H0)')


'''
#True, Prediction
#Y_true = data_0[:,0]

#Y_pred_0 = data_0[:,1]
#Y_pred_1 = data_1[:,1]

###### Statistical Hypothesis Testing #####
tb = mcnemar_table(y_target=Y_true, 
                   y_model1=np.array(Y_pred_0), 
                   y_model2=np.array(Y_pred_1)) 
print(tb)

#chi2, p = mcnemar(ary=tb, exact=True, corrected=True)
chi2, p = mcnemar(ary=tb, exact=False, corrected=True)

print('chi-squared:', chi2)
#print('p-value: %4f' % p)
print('p-value: ', p)

# interpret the p-value
alpha = 0.05
if p > alpha:
	print('Same proportions of errors (fail to reject H0)')
else:
	print('Different proportions of errors (reject H0)')
'''


