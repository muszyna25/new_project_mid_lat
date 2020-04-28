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
from scipy.stats import wilcoxon
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

# e.g., /global/cscratch1/sd/muszyng/cnn_model_runs/Model_A/Model_A_0/27149319
# ... hyper_params_27149319.csv
# ... model_x.h5
# ... training_x.log

path='/global/cscratch1/sd/muszyng/cnn_model_runs/'
arch_names = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
cv_rounds = 10
n_models = 5

'''
for a in arch_names:
    #checkpoint_dir = os.path.join(os.environ['SCRATCH'],'cnn_model_runs/%s/%s/%i' % (model_name[:-2], model_name, job_id))
    for cvr in range(0,cv_rounds):
        for n in range(0,n_models):
'''

model = load_model(path + 'Model_A/Model_A_1/27180150/model_2.h5')
#model = load_model('/global/cscratch1/sd/muszyng/cnn_model_runs/sh/27322379/model_0.h5')

X = np.empty([500,60,120,40])
Y = np.zeros((500))

model.summary()

y = model.predict(X)
print(y.argmax(axis=-1))
y_pred =y.argmax(axis=-1)
# or
print(np.argmax(y,axis=1))

#y_pred = np.expand_dims(y_pred, axis=1)

print('Shape of y_pred', y_pred.shape)
print(len(y_pred.shape),len(Y.shape))

'''
###### Statistical Hypothesis Testing #####
tb = mcnemar_table(y_target=Y, 
                   y_model1=np.array(y_pred), 
                   y_model2=np.array(y_pred)) 
print(tb)

chi2, p = mcnemar(ary=tb, exact=True, corrected=True)

print('chi-squared:', chi2)
print('p-value:', p)
'''


