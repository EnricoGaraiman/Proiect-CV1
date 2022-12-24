from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def svm_classification(train_features, labels_train, test_features, labels_test):
    # set parameters for SWM
    clf = svm.SVC(C=1, kernel='rbf', verbose=False, gamma='scale')

    # train step
    clf.fit(train_features, labels_train)

    # prediction
    predictions = clf.predict(test_features)

    acc = accuracy_score(labels_test, predictions)
    print('Prediction accuracy SVM:', acc * 100, '%')

    # confusion matrix display
    cmx = confusion_matrix(labels_test, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Confusion matrix', fontsize=20)
    plt.xlabel('True label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    disp.plot(ax=ax)
    # plt.show()
    plt.savefig('results/confusion-matrix-svm.png')

    print('SVM classification done')


def knn_classification(train_features, labels_train, test_features, labels_test):
    # KNN
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, algorithm='brute', weights='uniform')
    classifier.fit(train_features, labels_train)

    # predict
    predictions = classifier.predict(test_features)

    acc = accuracy_score(labels_test, predictions)
    print('Prediction accuracy KNN:', acc * 100, '%')

    # confusion matrix display
    cmx = confusion_matrix(labels_test, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Confusion matrix', fontsize=20)
    plt.xlabel('True label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    disp.plot(ax=ax)
    # plt.show()
    plt.savefig('results/confusion-matrix-knn.png')

    print('KNN classification done')


def kmeans_classification(train_features, labels_train, test_features, labels_test):
    # K Means
    classifier = KMeans(n_clusters=4, n_init=3, algorithm='lloyd', verbose=False)
    classifier.fit(train_features)

    # predict train & test
    predictions_train = classifier.predict(train_features)
    predictions_test = classifier.predict(test_features)

    acc = accuracy_score(labels_train, predictions_train)
    print('Prediction accuracy K Means (train):', acc * 100, '%')

    acc = accuracy_score(labels_test, predictions_test)
    print('Prediction accuracy K Means (test):', acc * 100, '%')

    # confusion matrix display
    cmx = confusion_matrix(labels_test, predictions_test, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Confusion matrix', fontsize=20)
    plt.xlabel('True label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    disp.plot(ax=ax)
    # plt.show()
    plt.savefig('results/confusion-matrix-kmeans.png')

    print('K Means classification done')
