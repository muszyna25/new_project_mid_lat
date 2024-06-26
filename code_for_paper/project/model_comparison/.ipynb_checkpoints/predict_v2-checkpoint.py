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
        X=h5f['X'][:]
        Y=h5f['Y'][:]
        print(' done',X.shape, Y.shape)
 
    return X, Y

###### MAIN ##########

l_models = read_results_from_csv('all_best_models.csv')
#print(l_models)
print(l_models[1])

start_ind = 401 
n_files = 20
path = '/global/cscratch1/sd/muszyng/ethz_data/project_atmo_block_datasets/generated_datasets/'
input_dir = path + 'data_classification_regression/data_for_paper/all_data_v2/source_d/'
file_list = 'file_list.txt'

counter=1
for i in l_models:
    print(i[0])
    print(i[2] + '/model_' + str(i[1]) + '.h5')
    model = load_model(i[2] + '/model_' + str(i[1]) + '.h5')
    n_params = model.count_params()
    print(n_params)
    model.summary()

    # Read test data
    # from 401 to 420
    lf_test = get_train_val_names_v2(file_list, start_ind, n_files)
    print(lf_test)

    if counter == 10:
        start_ind = 401 
        counter=1
    else:
        start_ind = start_ind + n_files 
        counter+=1

    #start_ind = start_ind + n_files 

    X_all = np.empty((0, 60, 120, 40))
    Y_all = np.empty((0,1))

    #for j in lf_test[0:1]:
    for j in lf_test:
        X, Y = read_hdf5_file(input_dir + j)
        print(X.shape)
        X_all = np.concatenate([X_all,X])
        Y_all = np.concatenate([Y_all,Y])

    print('X_all: ', X_all.shape)
    num_zeros = (Y_all == 0).sum()
    num_ones = (Y_all == 1).sum()
    print('Y ', num_zeros, num_ones)

    scores = model.evaluate(X_all, Y_all, verbose=0)
    print('scores: ', scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    np.savetxt('classifier_predict/' + 'loss_acc_' + i[0] + '.csv', scores, fmt='%.5f')

    y = model.predict(X_all,batch_size=256)
    print(y)
    p_new = np.array([1.0 - i for i in y])
    y_new = np.concatenate([y, p_new], axis=1)
    y_pred = y_new.argmax(axis=-1)
    y_pred_new = [1 if i == 0 else 0 for i in y_pred]
    print(y_pred_new[29:33], Y_all[29:33].flatten())

    report = classification_report(Y_all, y_pred_new, output_dict=True)
    print(report)
    print('ACC: ', accuracy_score(Y_all, y_pred_new))
    #np.savetxt('classifier_predict/' + 'report_' + i[0] + '.csv', report)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(r'classifier_predict/' + 'report_' + i[0] + '.csv')

    # true, pred 
    dat = np.array([Y_all.flatten(), y_pred_new])
    dat = dat.T
    np.savetxt('classifier_predict/' + i[0] + '.csv', dat, fmt='%i,%i')




