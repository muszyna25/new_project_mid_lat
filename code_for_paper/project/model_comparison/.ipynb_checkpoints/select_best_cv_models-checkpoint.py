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

def read_results_from_csv(fname):
    input_file = csv.DictReader(open(fname))
    my_list = []
    for row in input_file:
        my_list.append(row)
    return my_list

######################################
def save_to_csv_file(fname,a,b,c,d,e,f,g):
    file_exists = os.path.isfile(fname)
    #file_exists = os.path.isfile(fname + '.csv')
    with open(fname, mode='a') as csv_file: 
        fd = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        writer = csv.DictWriter(csv_file, fieldnames=fd)
        if not file_exists:
            writer.writeheader()
        D = {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'g': g}
        writer.writerow(D)

################## MAIN ###############

# e.g., /global/cscratch1/sd/muszyng/cnn_model_runs/Model_A/Model_A_0/27149319
path='/global/cscratch1/sd/muszyng/cnn_model_runs/'

arch_names = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
cv_rounds = 10
n_models = 5

'''
for a in arch_names:
    for cvr in range(0,cv_rounds):
        for n in range(0,n_models):
            checkpoint_dir = os.path.join(os.environ['SCRATCH'],'cnn_model_runs/%s/%s_%i/' % (a,a,cvr))
            dirs = [os.path.basename(x) for x in glob.glob(checkpoint_dir + '*')]
            if dirs != []: # This condition is not needed and can be removed when I executed all architectures.
                cv_round_dir = checkpoint_dir + dirs[0]
                print(a, cvr, n, checkpoint_dir, dirs, cv_round_dir)
                mod_list = [os.path.basename(x) for x in glob.glob(cv_round_dir + '/hyper*')]
                print(mod_list)
                data = read_results_from_csv(cv_round_dir + '/' + mod_list[0])
'''

for a in arch_names:
    for cvr in range(0,cv_rounds):
        checkpoint_dir = os.path.join(os.environ['SCRATCH'],'cnn_model_runs/%s/%s_%i/' % (a,a,cvr))
        dirs = [os.path.basename(x) for x in glob.glob(checkpoint_dir + '*')]
        if dirs != []: # This condition is not needed and can be removed when I executed all architectures.
            for d in dirs:
                cv_round_dir = checkpoint_dir + d
                l_trains = [os.path.basename(x) for x in glob.glob(cv_round_dir + '/train*')]
                print('CV round: ', cv_round_dir)
                l_trains_sorted = sorted(l_trains,  key = lambda x:x.split('_')[1][0])
                print('List of training.log files: ', l_trains_sorted)
                for f in l_trains_sorted:
                    i_model = f.split('_')[1][0]
                    data = read_results_from_csv(cv_round_dir + '/' + f)
                    last_train_acc = data[len(data)-1]['acc']
                    last_val_acc = data[len(data)-1]['val_acc']
                    if float(last_val_acc) >= 0.70 and float(last_train_acc) >= 0.70: 
                        #model = load_model(cv_round_dir + '/model_' + str(i_model) + '.h5')
                        #n_params = model.count_params()
                        #print(n_params)
                        print('Val ACC: ', last_val_acc)
                        save_to_csv_file('file.csv', a, a + '_' + str(cvr), a, i_model, last_train_acc, last_val_acc, cv_round_dir)

                #for f in l_trains_sorted:
                #    fd  = open(f, 'r')
                    

                #cv_round_dir = checkpoint_dir + dirs[0]
                #print(a, cvr, n, checkpoint_dir, dirs, cv_round_dir)
                #mod_list = [os.path.basename(x) for x in glob.glob(cv_round_dir + '/hyper*')]
                #print(mod_list)
                #data = read_results_from_csv(cv_round_dir + '/' + mod_list[0])

                #print(data[4]['conv_sizes'])
                #df = pd.read_csv(cv_round_dir + '/' + mod_list[0])
                #pd.options.display.max_columns = len(df.columns)
                #print(df)

                #last_train_acc = data[len(data)-1]['loss']
                #last_val_acc = data[len(data)-1]['val_loss']




