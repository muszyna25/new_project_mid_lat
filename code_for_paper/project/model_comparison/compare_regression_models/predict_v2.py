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
from sklearn.metrics import explained_variance_score
from tensorflow.python.keras import backend as K

def read_results_from_csv(fname):
    #input_file = csv.DictReader(open(fname))
    with open(fname) as csvfile:
        input_file = csv.reader(csvfile, delimiter=',')
        next(input_file, None)
        my_list = []
        for row in input_file:
            my_list.append(row)
    
    return my_list

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
        #Y=h5f['Y'][:]
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

counter=1
for i in l_models:
    print(i[0])
    print(i[2] + '/model_' + str(i[1]) + '.h5')
    model = load_model(i[2] + '/model_' + str(i[1]) + '.h5', custom_objects=dependencies)
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
    Y_all = np.empty((0,3))

    #for j in lf_test[0:2]:
    for j in lf_test:
        X, Y = read_hdf5_file(input_dir + j)
        print(X.shape)
        X_all = np.concatenate([X_all,X])
        Y_all = np.concatenate([Y_all,Y])

    scores = model.evaluate(X_all, Y_all, verbose=0)
    print('scores: ', scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

    y = model.predict(X_all,batch_size=256)
    print(y[:10])
    print(y.shape)

    s = lins_ccc(Y_all, y)
    print('CCC: ', s)

    print("names:", model.metrics_names)
    np.savetxt('regressor_predict/' + 'loss_ccc_' + i[0] + '.csv', scores, fmt='%.5f')

    # true, pred 
    save_to_hdf5_format(Y_all, y, 'regressor_predict/' + i[0] + '.h5')

    #dat = np.array([Y_all, y])
    #dat = dat.T

    #df = pandas.DataFrame(dat).transpose()
    #df.to_csv(r'regressor_predict/' + i[0] + '.csv')

    #with open('regressor_predict/' + i[0] + '.csv', 'w') as fd:
    #    for slc in dat:
    #        np.savetxt(fd, slc, fmt='%i,%i')
    #np.savetxt('regressor_predict/' + i[0] + '.csv', dat, fmt='%i,%i')

'''
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
'''



