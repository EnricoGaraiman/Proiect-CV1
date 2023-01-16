from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def svm_classification(train_features, labels_train, test_features, labels_test):
    # set parameters for SWM
    clf = svm.SVC(
        C=1,
        kernel='rbf',
        verbose=False,
        gamma='scale'
    )

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
    classifier = KNeighborsClassifier(
        n_neighbors=3,
        metric='minkowski',
        p=2,
        algorithm='brute',
        weights='uniform'
    )
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

def mlp_classification(train_features, labels_train, test_features, labels_test):
    # MLP
    classifier = MLPClassifier(
        random_state=100,
        max_iter=300,
        hidden_layer_sizes=[2000, 800, 400, 40],
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        verbose=False,
        early_stopping=True
    )
    classifier.fit(train_features, labels_train)

    # predict
    predictions = classifier.predict(test_features)

    acc = accuracy_score(labels_test, predictions)
    print('Prediction accuracy MLP:', acc * 100, '%')

    # confusion matrix display
    cmx = confusion_matrix(labels_test, predictions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmx, display_labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Confusion matrix', fontsize=20)
    plt.xlabel('True label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    disp.plot(ax=ax)
    # plt.show()
    plt.savefig('results/confusion-matrix-mlp.png')

    print('MLP classification done')
