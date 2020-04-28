#!/usr/bin/env python

import sys 
import os


l_train = []
l_val = []
l_test = []

for i in range(0,10):
    for j in range(0,20):
        train_pattern = str(i) +  "_" + "train" + "_" + str(j) + "_.h5"
        val_pattern = str(i) +  "_" + "val" + "_" + str(j) + "_.h5"
        test_pattern = str(i) +  "_" + "test" + "_" + str(j) + "_.h5"
        l_train.append(train_pattern)
        l_val.append(val_pattern)
        l_test.append(test_pattern)

l_all = [l_train, l_val, l_test] 
l_fns = ["train", "val", "test"]

#for l,fn in zip(l_all,l_fns):
for l in l_all:
    with open('all_fold_files' + '.txt', 'a+') as filehandle:
        for listitem in l:
            filehandle.write('%s\n' % listitem)


