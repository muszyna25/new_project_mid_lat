import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt

def normalize_data(X, Y):
    X = X.reshape(-1,1)
    print(X)
    Y_POS = Y == 1
    print(Y_POS)
    _X = X[Y_POS]
    
    N = (2.0 * (_X - min(_X))/(max(_X) - min(_X))) - 1
    
    counter = 0
    X_n = X.astype(float)
    for i in range(0,Y_POS.shape[0]):
        if Y_POS[i] == True:
            print(i)
            print(N[counter])
            X_n[i] = N[counter]
            counter+=1
    
    return np.array(X_n)
    
################ MAIN #########

with h5py.File(sys.argv[1], 'r') as hf:
    Z=hf['Z'][:]
    
print('Z: ', Z[200:1000])
print('Z: ', Z.shape)
'''
inds_left = np.argwhere((Z[:,2] >= 0.0))
Z = np.delete(Z, inds_left, axis=0)
inds_right = np.argwhere((Z[:,2] <= -0.85))
Z = np.delete(Z, inds_right, axis=0)
print('Z : ', Z.shape)
'''
'''
inds_left = np.argwhere((Z[:,2] == 0.0))
Z = np.delete(Z, inds_left, axis=0)

inds_left = np.argwhere((Z[:,0] <= 5.0))
Z = np.delete(Z, inds_left, axis=0)
inds_left = np.argwhere((Z[:,0] >= 115.0))
Z = np.delete(Z, inds_left, axis=0)
inds_right = np.argwhere((Z[:,2] <= 10.0))
Z = np.delete(Z, inds_right, axis=0)
inds_right = np.argwhere((Z[:,2] >= 40.0))
Z = np.delete(Z, inds_right, axis=0)
print('Z : ', Z.shape)
'''
fig, axs = plt.subplots(3, 1, figsize=(18, 8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.5, wspace=.001)
axs = axs.ravel()

#n_bins = np.linspace(0., 120., 25)
n_bins = np.linspace(-1.0, 1.0, 30)

axs[0].hist(Z[:,0], bins=n_bins)
#axs[0].set_yscale('log')
axs[0].set_title('Mass center: x')
axs[1].hist(Z[:,1], bins=n_bins)
#axs[1].set_yscale('log')
axs[1].set_title('Mass center: y')
axs[2].hist(Z[:,2], bins=n_bins)
#axs[2].set_yscale('log')
axs[2].set_title('Radius: r')

#plt.yscale('log', nonposy='clip')

plt.show()
   
print('Z: ', Z.shape)

    
    
    
    
    
    
    
    

'''
with h5py.File(sys.argv[1], 'r') as hf:
    Y=hf['Y'][:]
    Z=hf['Z'][:]

print('Z shape: ', Z.shape[1])

Z = Z.astype(float)
for i in range(0, Z.shape[1]):
    A = Z[:,i]
    Z[:,[i]] = normalize_data(A,Y)

print(Z)
'''

'''
with h5py.File(sys.argv[1], 'r') as hf:
    X=hf['X'][:]
    Y=hf['Y'][:]

Y_tmp = Y == 1

print('Y_tmp', type(Y_tmp), Y_tmp.shape, Y_tmp[0:10])
            
X_n = np.array(X)
print('X', type(X), X.shape)

#I = [i for i in range(0, Y_tmp.shape[0]) if Y_tmp[i] == True]
I = [i for i in range(0, Y_tmp.shape[0]) if Y[i] == 1]
#print('I', type(I), I.shape, I[0:10])
print('I', type(I), I[0:10], np.array(I).shape[0])

A=X_n[I,:,:,:]
print('A', A.shape, A[0:2])
'''






