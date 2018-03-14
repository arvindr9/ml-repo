import sys

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import scipy
count = .10
epsilon = .0013
#fun = L_2(T(x + e) - T(x))
#X_adv = scipy.optimize.minimize(fun, 784, method=COBYLA, constraints = cons)

#train model to create a cl
# for each x:
#     define fun to be L_2(T(x + e) - T(x))
#     find X + e such that |e| <= epsilon and fun is minimal
#     append X + e to adversarial list
#

def find_adv(X, clf, sign):
    T_X = clf.predict_proba(X)
    #X = X.reshape(-1, 1)
    fun = lambda x: sign * np.sum((clf.predict_proba(np.array(x).reshape(1, -1)) - T_X)**2)
    cons = ({'type': 'ineq',
            'fun': lambda x: np.sum(epsilon**2 - (np.array(x).reshape(1, -1)- X)**2)})
    print(count)
    print(X.shape)
    obj = scipy.optimize.minimize(fun, list(X[0]), method='COBYLA', constraints = cons)
    return obj

def find_adv2(X, clf, sign):
    T_X = clf.predict_proba(X)
    #X = X.reshape(-1, 1)
    fun = lambda x: sign * np.sum((clf.predict_proba(x.reshape(1, -1)) - T_X)**2)
    cons = ({'type': 'ineq',
            'fun': lambda x: np.sum(epsilon**2 - (x.reshape(1, -1)- X)**2)})
    print(count)
    print(X.shape)
    obj = scipy.optimize.minimize(fun, X[0], method='COBYLA', constraints = cons)
    return obj
def adversarialize(X, clf, sign = -1):
    j = 0
    X_adv = []
    for i in range(X.shape[0]):
        if j == 0:
            X_adv = find_adv(X[0].reshape(1, -1), clf, sign)
            j += 1
            continue
        Xi_adv = find_adv(X[i].reshape(1, -1), clf, sign)
        X_adv = np.concatenate((X_adv, Xi_adv))
    return X_adv

def createRandomForest(X, y):
    clf = RFC(max_depth=65)
    clf.fit(X, y)
    return clf

def extract(mnist):
    train, test = mnist.train, mnist.test
    return train.images, train.labels, test.images, test.labels

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    X_train, y_train, X_test, y_test = extract(mnist)
    print(X_train.shape)
    clf = createRandomForest(X_train, y_train)
    X_train_adv = adversarialize(X_train, clf)


if __name__ == '__main__':
    main()