import numpy as np

def normalize_regres_data(Z, Y, c_min=None, c_max=None):

    Y_POS = Y == 1

    Z1 = Z[:,0].reshape(-1,1)
    Z_d = Z1[Y_POS]
    print(Z.shape, Z1.shape, Z_d.shape)

    l_consts = []
    X_n = np.empty((Z_d.shape[0], 3)) 
    for j in range(0,3):
        X = Z[:,j].reshape(-1,1)
        _X = X[Y_POS]
        print("_X: ", _X) 

        if c_min is None and c_max is None:
            N = (2.0 * (_X - min(_X))/(max(_X) - min(_X))) - 1.0 
            l_consts.append([min(_X), max(_X)])
        else:
            N = (2.0 * (_X - c_min)/(c_max - c_min)) - 1.0 

        X_n[:,j] = N 

    return np.array(X_n), l_consts 

########################33

A = np.empty((10,3))
print(A.shape)

B = np.array([1,0,1,0,1,0,0,0,0,1]).reshape(-1,1)
print(B.shape)

normalize_regres_data(A,B)
