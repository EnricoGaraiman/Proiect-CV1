import random as random
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import src.helpers as helpers
import glob
import sys
import numpy as np

def get_class_by_path(path):
    # get class index based on path name
    return int(path.split('/')[1].split('\\')[0])


def get_train_and_test_dataset(dataset_dir, percent, scale_factor):
    dataset_train = []
    dataset_test = []
    labels_train = []
    labels_test = []

    # get random indexes
    random.seed(101)
    indexes = random.sample(range(0, 9), percent)

    # read all data and split in train and set based on random indexes
    for class_index in range(len(glob.glob(dataset_dir + '/*'))):
        for index, path in enumerate(glob.glob(dataset_dir + '/' + str(class_index) + '/*')):
            img_original = io.imread(path)

            # img_original = helpers.rgb2gray(img_original)
            new_h = int(np.shape(img_original)[0] * scale_factor)
            new_w = int(np.shape(img_original)[1] * scale_factor)
            img_original = resize(img_original, [new_h, new_w],  anti_aliasing=True)

            if index in indexes:
                dataset_train.append(img_original)
                labels_train.append(get_class_by_path(path))
            else:
                dataset_test.append(img_original)
                labels_test.append(get_class_by_path(path))

    # return train & test dataset, train & test labels
    print('Data loaded')
    return dataset_train, labels_train, dataset_test, labels_test, len(glob.glob(dataset_dir + '/*'))


def dataset_visualisation(dataset_train, labels_train, dataset_test, labels_test, number_classes, percent, fig_name_suffix = '', show=True, save=True):
    # train
    fig, ax = plt.subplots(number_classes, percent, figsize=(10, 5))
    for i in range(number_classes):
        for j in range(percent):
            ax[i][j].imshow(dataset_train[i * percent + j], cmap='gray')
            ax[i][j].set_title(labels_train[i * percent + j])

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig('results/training_data_visualisation' + fig_name_suffix + '.jpg')

    # test
    fig, ax = plt.subplots(number_classes, 10 - percent, figsize=(10, 5))

    for i in range(number_classes):
        for j in range(10 - percent):
            ax[i][j].imshow(dataset_test[i * (10 - percent) + j], cmap='gray')
            ax[i][j].set_title(labels_test[i * (10 - percent) + j])

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        fig.savefig('results/testing_data_visualisation' + fig_name_suffix + '.jpg')


def dataset_mask(dataset_train, dataset_test, histogram = False):

    # test image
    if histogram:
        image = dataset_train[0]
        fig = plt.figure()
        plt.imshow(image)
        plt.show()
        fig.savefig('results/first_image_for_histogram.jpg')

        fig = plt.figure()
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        red_pixels = red.flatten()
        green_pixels = green.flatten()
        blue_pixels = blue.flatten()

        plt.hist(red_pixels, bins=256, density=False, color='red', alpha=0.5)
        plt.hist(green_pixels, bins=256, density=False, color='green', alpha=0.4)
        plt.hist(blue_pixels, bins=256, density=False, color='blue', alpha=0.3)

        plt.xticks(ticks=np.linspace(0, 1, 17), labels=range(0, 257, 16))
        plt.title("Color Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
        plt.show()
        fig.savefig('results/histogram_first_image_train.jpg')

    # get mask data for train and test
    dataset_train_mask = list(map(helpers.hand_mask, dataset_train))
    dataset_test_mask = list(map(helpers.hand_mask, dataset_test))

    # return mask dataset train & test
    print('Mask done')
    return dataset_train_mask, dataset_test_mask


def dataset_contours(dataset_train_mask, dataset_test_mask, dataset_train, dataset_test):

    # get contours for masked dataset and for original dataset
    train_packed = list(map(helpers.hand_contour, dataset_train_mask, dataset_train))
    test_packed = list(map(helpers.hand_contour, dataset_test_mask, dataset_test))

    # unpack
    dataset_test_mask_contour = [i[0] for i in test_packed]
    dataset_test_contour = [i[1] for i in test_packed]
    dataset_train_mask_contour = [i[0] for i in train_packed]
    dataset_train_contour = [i[1] for i in train_packed]

    # return contours for masked dataset and for original dataset
    print('Contours done')
    return dataset_train_mask_contour, dataset_train_contour, dataset_test_mask_contour, dataset_test_contour

