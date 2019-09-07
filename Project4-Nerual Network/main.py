import pickle
import numpy as np
import scipy.io as sio

# run this to save edit best_parameters.pickle which will be used to determine performance on the autograder
# Also feel free to use this file to do any testing as it will not be called by the autograder

bostonData = sio.loadmat('./boston.mat')
xTr = bostonData['xTr']
yTr = bostonData['yTr']
#print(xTr)
d, _ = np.shape(xTr)
#print(d)


best_parameters = {
    'TRANSNAME' : 'sigmoid',
    'ROUNDS' : 13,
    'ITER' : 100,
    'STEPSIZE' : 0.01,
    'wst' : np.array([1,20,30,20,13])
}

with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_parameters, f)