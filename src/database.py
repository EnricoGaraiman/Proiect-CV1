import random as random
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import src.helpers as helpers
import glob
import sys
import numpy as np

def get_class_by_path(path):
    return path.split('/')[1].split('\\')[0]

def get_train_and_test_dataset(dataset_dir, percent, scale_factor):
    dataset_train = []
    dataset_test = []
    labels_train = []
    labels_test = []

    random.seed(42)
    indexes = random.sample(range(0, 9), percent)

    for class_index in range(len(glob.glob(dataset_dir + '/*'))):
        for index, path in enumerate(glob.glob(dataset_dir + '/' + str(class_index) + '/*')):
            img_original = io.imread(path)
            img_original = helpers.rgb2gray(img_original)
            new_h = int(np.shape(img_original)[0] * scale_factor)
            new_w = int(np.shape(img_original)[1] * scale_factor)
            if index in indexes:
                dataset_train.append(resize(img_original, [new_h, new_w], anti_aliasing=True))
                labels_train.append(get_class_by_path(path))
            else:
                dataset_test.append(resize(img_original, [new_h, new_w], anti_aliasing=True))
                labels_test.append(get_class_by_path(path))

    return dataset_train, labels_train, dataset_test, labels_test, len(glob.glob(dataset_dir + '/*'))


def dataset_visualisation(dataset_train, labels_train, dataset_test, labels_test, number_classes, percent, show=True, save=True):
    # train
    fig, ax = plt.subplots(number_classes, percent, sharex='col', sharey='row', figsize=(10, 5))

    for i in range(number_classes):
        for j in range(percent):
            ax[i][j].imshow(dataset_train[i * percent + j], cmap='gray')
            ax[i][j].set_title(labels_train[i * percent + j])

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig('results/training_data_visualisation.jpg')

    # test
    fig, ax = plt.subplots(number_classes, 10-percent, sharex='col', sharey='row', figsize=(10, 5))

    for i in range(number_classes):
        for j in range(10-percent):
            ax[i][j].imshow(dataset_test[i * (10-percent) + j], cmap='gray')
            ax[i][j].set_title(labels_test[i * (10-percent) + j])

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig('results/testing_data_visualisation.jpg')