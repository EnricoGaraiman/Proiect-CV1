from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np


def svm_classification(train_features, labels_train):
    # set parameters for SWM
    clf = svm.SVC(C=1, kernel='rbf', verbose=True)

    # train step
    clf.fit(train_features, labels_train)

    # return clf
    print('SVM classification done')
    return clf


def prediction(clf, test_features, labels_test):
    predictions = clf.predict(test_features)

    acc = accuracy_score(labels_test, predictions)

    print('Prediction accuracy:', acc * 100, '%')
    return predictions, acc

# + NN