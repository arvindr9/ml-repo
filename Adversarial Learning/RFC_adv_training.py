from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from cleverhans.attacks import FastGradientMethod
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier as RFC

epsilon = .013

def extract(mnist):
    train, test = mnist.train, mnist.test
    return train.images, train.labels, test.images, test.labels

def createRandomForest(X, y):
    clf = RFC(max_depth=65)
    clf.fit(X, y)
    return clf

def adversarialize(X):
    X_adv = X + epsilon
    return X_adv

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, y_train, X_test, y_test = extract(mnist)
    clf = createRandomForest(X_train, y_train)
    X_train_adv = adversarialize(X_train)
    X_test_adv = adversarialize(X_test)
    print("Original results:")
    print(f"\tTrain accuracy: {clf.score(X_train, y_train)}")
    print(f"\tTest accuracy: {clf.score(X_test, y_test)}")
    print(f"\tAdversarial example accuracy (generated from the test set): {clf.score(X_test_adv, y_test)}")
    m = X_train.shape[0]
    indices = np.random.permutation(m)
    X_t2, X_adv2 = X_train[indices[:int(m/2)]], X_train_adv[indices[int(m/2):]]
    y_t2, y_adv2 = y_train[indices[:int(m/2)]], y_train[indices[int(m/2):]]
    X_train_mixed = np.concatenate((X_t2, X_adv2))
    y_train_mixed = np.concatenate((y_t2, y_adv2))
    clf2 = createRandomForest(X_train_mixed, y_train_mixed)
    print(f"Results after adversarial training:")
    print(f"\tTest accuracy: {clf2.score(X_test, y_test)}")
    print(f"\tAdversarial example accuracy (generated from the test set): {clf2.score(X_test_adv, y_test)}")

if __name__ == '__main__':
    main()