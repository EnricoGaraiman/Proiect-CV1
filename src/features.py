from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.transform import resize

def extract_features_hog(dataset, width, height, example=False):
    images_hog = []
    features = []

    # for each image extract hog image
    for image in dataset:
        # set same dimensions and HOG will return same dimension of features
        image = resize(image, [height, width])

        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=-1)

        images_hog.append(hog_image)
        features.append(fd)

    # see example for one image
    if example:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.axis('off')
        ax1.imshow(dataset[0], cmap='gray')
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(images_hog[0], in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
        fig.savefig('results/hog_hands_example.jpg')

    print('HOG done')
    return features, images_hog
