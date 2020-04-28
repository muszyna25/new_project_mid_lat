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
from statistics import mean, stdev

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

def lins_ccc(y_true, y_pred):
    t = y_true.mean()
    p = y_pred.mean()
    St = y_true.var()
    Sp = y_pred.var()
    Spt = np.mean((y_true - t) * (y_pred - p))
    return 2 * Spt / (St + Sp + (t - p)**2)

def save_to_hdf5_format(Y_true, Y_pred, fname):

    print('Data shapes: ', Y_true, Y_pred)

    # creat new file
    fn = fname
    print('Create %s' %fn)
    hf = h5py.File(fn, 'w')
    hf.create_dataset('Y_t', data=Y_true)
    hf.create_dataset('Y_p', data=Y_pred)
    hf.close()
    print('[+] done...!')

###### MAIN ##########

path = 'regressor_predict/'

l_models = read_results_from_csv('all_best_models.csv')
print(l_models[1])

counter=1
names = [l_models[i][0][:-2] for i in range(0,50) if i % 10 == 0]
print(names)
avg_accs = np.zeros((1, 5))
print(avg_accs)
avg_ACCs = dict(zip(names, avg_accs))

l_accs = []
l_ys_true = np.empty((0,3))
l_ys_pred = np.empty((0,3))

for i in l_models:
    print(i[0])

    #data = read_data_from_csv(path + i[0] + '.csv') 
    Y_true, Y_pred = read_hdf5_file(path + i[0] + '.h5')
    #print(data.shape)

    #True, Prediction
    #Y_true = data[:,0:1]
    #Y_pred = data[:,1:]
    print(Y_true.shape, Y_pred.shape)
    #acc = accuracy_score(Y_true, Y_pred)
    acc = lins_ccc(Y_true, Y_pred)
    l_accs.append(acc)
    print('CCC: ', acc)
    l_ys_true = np.concatenate((l_ys_true, Y_true), axis=0)
    l_ys_pred = np.concatenate((l_ys_pred, Y_pred), axis=0)
    print('l_ys_* ', l_ys_true.shape, l_ys_pred.shape)

    if counter == 10:
        #np.savetxt(path + i[0][:-2] + '.csv', l_ys, fmt='%s,%s')
        save_to_hdf5_format(l_ys_true, l_ys_pred, 'regressor_predict/avgs/' + i[0][:-2] + '.h5')
        print('for mean: ', l_accs)
        avg_ACCs[i[0][:-2]] = [round(mean(l_accs),3), round(stdev(l_accs),3)]
        counter=1
        l_accs = [] 
        l_ys_true = np.empty((0,3))
        l_ys_pred = np.empty((0,3))
        #l_ys = np.empty((0,2))
    else:
        counter+=1

print(avg_ACCs)

df = pandas.DataFrame.from_dict(avg_ACCs, orient="index")
df.to_csv(r'regressor_predict/avgs/' + 'avg_performances.csv')




