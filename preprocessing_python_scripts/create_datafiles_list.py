#!/usr/bin/env python

import sys 
import os

################################################################################################
#   INPUT:
#       dir_path -- directory path
#       idx_year -- indices for sorting a list of files, e.g. ERA-Interim: (3,7) or ETHZ masks: (6, 10)
#       out_fname -- file name of output text file
#   OUTPUT:
#       output_fn -- output text file
################################################################################################

if __name__ == "__main__":
    
    if len(sys.argv[1:]) < 1:
        print("Provide directory path")
        sys.exit(0)

    dir_path = sys.argv[1]
    out_fname = sys.argv[2]
    output_fn = out_fname + '.txt'
    idx_year = (0,4) # i.e, ERAxxxx.nc or BLOCKSxxxx.nc
    #idx_year = (6,10) # i.e, ERAxxxx.nc or BLOCKSxxxx.nc

    #df_list = list(filter(lambda x: x.endswith('.nc'), os.listdir(dir_path)))
    df_list = list(filter(lambda x: x.endswith('.h5'), os.listdir(dir_path)))
    df_list_sorted = sorted(df_list,  key = lambda x:x[idx_year[0]:idx_year[1]])
    print(df_list_sorted)
    
    #Save list to text file.
    with open(output_fn, 'w') as filehandle:
        for listitem in df_list_sorted:
            filehandle.write('%s\n' % listitem)


