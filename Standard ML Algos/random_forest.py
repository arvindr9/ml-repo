import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier as RFC

def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print(type(mnist))
    X = mnist.train.images
    y = mnist.train.labels
    clf = RFC(max_depth=70)
    clf.fit(X, y)
    print(X.shape)
    print(clf.score(X, y))
    print(clf.score(mnist.test.images, mnist.test.labels))

if __name__ == '__main__':
    main()