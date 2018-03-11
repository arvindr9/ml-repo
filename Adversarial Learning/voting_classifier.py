from tensorflow.examples.tutorials.mnist import input_data
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC, VotingClassifier
import numpy as np


def main():
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=False)
    clf1 = LR()
    clf2 = RFC()
    clf3 = GNB()
    eclf = VotingClassifier(estimators = [
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    X = mnist.train.images
    y = mnist.train.labels
    print('starting')
    eclf = eclf.fit(X, y)
    print(eclf.score(X, y), eclf.score(mnist.test.images, mnist.test.labels))



if __name__ == '__main__':
    main()
