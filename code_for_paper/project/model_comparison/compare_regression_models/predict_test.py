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
from tensorflow.python.keras import backend as K
import pandas
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

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
        Y=h5f['Z'][:]
        print(' done',X.shape, Y.shape)
 
    return X, Y



def CCC(y_true, y_pred):
    # covariance between y_true and y_pred
    n_y_true = y_true - K.mean(y_true[:])
    n_y_pred = y_pred - K.mean(y_pred[:])
    s_xy = K.mean(n_y_true * n_y_pred)

    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)

    # variances
    s_x_sq = K.mean(K.pow(n_y_true,2))
    s_y_sq = K.mean(K.pow(n_y_pred,2))

    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)

    return ccc 

def CC(y_true, y_pred):

    n_y_true = (y_true - K.mean(y_true[:]))
    n_y_pred = (y_pred - K.mean(y_pred[:]))

    top = K.sum(n_y_true[:] * n_y_pred[:])
    bottom = K.sqrt(K.sum(K.pow(n_y_true[:],2))) * K.sqrt(K.sum(K.pow(n_y_pred[:],2)))

    result = top/bottom

    return result

def cc0(true,pred):

    index = 0
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CC(true,pred)

def cc1(true,pred):

    index = 1
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CC(true,pred)

def cc2(true,pred):

    index = 2
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CC(true,pred)

def ccc0(true,pred):

    index = 0
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CCC(true,pred)

def ccc1(true,pred):

    index = 1
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CCC(true,pred)

def ccc2(true,pred):

    index = 2
    #get only the desired class
    true = true[:,index]
    pred = pred[:,index]

    #return dice per class
    return CCC(true,pred)

def lins_ccc(y_true, y_pred):
    t = y_true.mean()
    p = y_pred.mean()
    St = y_true.var()
    Sp = y_pred.var()
    Spt = np.mean((y_true - t) * (y_pred - p))
    return 2 * Spt / (St + Sp + (t - p)**2)

dependencies = {
    'CCC': CCC,
    'CC': CC,
    'cc0': cc0,
    'cc1': cc1,
    'cc2': cc2,
    'ccc0': ccc0,
    'ccc1': ccc1,
    'ccc2': ccc2
}

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

###### MAIN ##########

l_models = read_results_from_csv('all_best_models.csv')
#print(l_models)
print(l_models[1])

start_ind = 401 
n_files = 20
path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/'
#input_dir = path + 'data_classification_regression/data_for_paper/all_data_v2/source_d/'
input_dir = path + 'data_classification_regression/data_for_paper/all_data_v2/out_regres_data/'
file_list = 'file_list.txt'

model = load_model('/global/cscratch1/sd/muszyng/cnn_model_runs/regression/Model_A/Model_A_0/28257709/model_' + str(2) + '.h5', custom_objects=dependencies)
#model = load_model('/global/cscratch1/sd/muszyng/cnn_model_runs/regression/Model_B/Model_B_0/28257721/model_' + str(2) + '.h5', custom_objects=dependencies)

n_params = model.count_params()
print(n_params)
model.summary()

# Read test data
# from 401 to 420
lf_test = get_train_val_names_v2(file_list, start_ind, n_files)
print(lf_test)

X_all = np.empty((0, 60, 120, 40))
Y_all = np.empty((0,3))

for j in lf_test:

    X, Y = read_hdf5_file(input_dir + j)
    print(X.shape)
    X_all = np.concatenate([X_all,X])
    Y_all = np.concatenate([Y_all,Y])

print('X_all: ', X_all.shape)
print('Z ', Y_all.shape)

scores = model.evaluate(X_all, Y_all, verbose=0)
print('score names: ', model.metrics_names)
print('scores: ', scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

y = model.predict(X_all,batch_size=256)
print(y[:10])
print(y.shape)

s = lins_ccc(Y_all, y)
print('CCC: ', s)

evs = explained_variance_score(Y_all, y, multioutput='uniform_average')
print('EVS: ', evs)

max_resid_error_0 = max_error(Y_all[:,0], y[:,0])
print('Max resid error 0: ', max_resid_error_0)

max_resid_error_1 = max_error(Y_all[:,1], y[:,1])
print('Max resid error 1: ', max_resid_error_1)

max_resid_error_2 = max_error(Y_all[:,2], y[:,2])
print('Max resid error 2: ', max_resid_error_2)

l_res = []
for i in range(Y_all.shape[0]):
    res = abs(Y_all[i,0] - y[i,0])
    l_res.append(res)

print('max res:', max(l_res))
arg_max = argmax(l_res)
print('argmax res:', arg_max, l_res[arg_max], Y_all[arg_max,0], y[arg_max,0])


