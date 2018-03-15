import numpy as np
from utils import *
import sklearn
f_train, f_test, f_adv_train, f_adv_test = np.load('train.npz'), np.load('test.npz'), np.load('adv_train.npz'), np.load('adv_test.npz')
X_train, y_train, X_test, y_test, X_adv_train, X_adv_test = f_train['arr_0'], f_train['arr_1'], f_test['arr_0'], f_test['arr_1'], f_adv_train['arr_0'], f_adv_test['arr_0']

perm = np.random.permutation(55000)
indices = perm[:27501]
indices_adv = perm[27501:]
print(X_train.shape, X_adv_train.shape)
X_train_mixed = np.concatenate((X_train[indices][:], X_adv_train[indices_adv][:]))
y_train_mixed = np.concatenate((y_train[indices][:], y_train[indices_adv][:]))
print(X_train_mixed.shape)
clf = createRandomForest(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_train_mixed, y_train_mixed))
print(clf.score(X_adv_train, y_train))