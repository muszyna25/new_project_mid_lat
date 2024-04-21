#!/usr/bin/env python

import sys 
import os
import matplotlib.pyplot as plt
import numpy as np
import statistics

################################################################################################
#   INPUT:
#       fn_list -- text file with blob sizes for three calendar years (data files)
#       dir_path -- path location of ETHZ nc datafiles
#
#   OUTPUT:
#       output -- average, median, mode blob size
################################################################################################

debug_plot = True
sizes = []

# open file and read the content in a list
with open(sys.argv[1], 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        current = line[:-1]

        # add item to the list
        sizes.append(int(current))

avg = statistics.mean(sizes)
print('Avg size: ', avg)

med = statistics.median(sizes)
print('Med size: ', med)

mod = statistics.mode(sizes)
print('Med size: ', mod)

if debug_plot:
    n, bins, patches = plt.hist(x=sizes, bins=np.linspace(0,1000,500), color='#0504aa', alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
