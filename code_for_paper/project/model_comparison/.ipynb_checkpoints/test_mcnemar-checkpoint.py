import numpy as np
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

y_mod1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
y_mod2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])

#tb = mcnemar_table(y_target=y_true, 
#                   y_model1=y_mod1, 
#                   y_model2=y_mod2)

#print(tb)

tb = np.array([[9945, 25],
                 [15, 15]])

print(tb)

chi2, p = mcnemar(ary=tb, exact=True, corrected=True)

print('chi-squared:', chi2)
print('p-value:', p)


