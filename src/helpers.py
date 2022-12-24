import numpy as np
import cv2

rgb2gray = lambda img: np.uint8(0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.144 * img[:, :, 2])


def hand_mask(img):
    # return mask for image
    mask = img[..., 0] > 90 / 255
    mask[mask > 0] = 255

    return mask


def hand_contour(img, img_orig):
    # get contour for image
    contur, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get max area for contours
    arie_max = 0
    rect_max = np.array([0, 0, 0, 0])
    for i in contur:
        temp = cv2.boundingRect(i)
        if temp[2] * temp[3] > arie_max:  # temp[2]*temp[3] rect (rect = (x, y, w, h))
            arie_max = temp[2] * temp[3]
            rect_max = temp

    # crop and return mask image and original image
    return crop_img(img, rect_max), crop_img(img_orig, rect_max)


def crop_img(img, rect):
    return img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]


def get_dimensions_for_resize(dataset_train, dataset_test):
    min_w = 0
    min_h = 0

    for img in dataset_train:
        if np.shape(img)[1] > min_w:
            min_w = np.shape(img)[1]
            min_h = np.shape(img)[0]

    for img in dataset_test:
        if np.shape(img)[1] > min_w:
            min_w = np.shape(img)[1]
            min_h = np.shape(img)[0]

    print("Resize dims:", min_w, min_h)
    return min_w, min_h
