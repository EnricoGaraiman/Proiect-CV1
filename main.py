import src.database as database
import src.features as features
import src.classification as classification
import src.helpers as helpers

# ___________________________________________
# PARAMETERS
# ___________________________________________
data_dir = 'database'
scale_factor = 0.2
percent = 8

# ___________________________________________
# MAIN
# ___________________________________________
if __name__ == '__main__':
    # get data
    dataset_train, labels_train, dataset_test, labels_test, number_classes = database.get_train_and_test_dataset(data_dir, percent, scale_factor)
    # database.dataset_visualisation(dataset_train, labels_train, dataset_test, labels_test, number_classes, percent)

    # mask visualisation
    dataset_train_mask, dataset_test_mask = database.dataset_mask(dataset_train, dataset_test)
    # database.dataset_visualisation(dataset_train_mask, labels_train, dataset_test_mask, labels_test, number_classes, percent, '_mask')

    # find contours and crop
    dataset_train_mask_contour, dataset_train_contour, dataset_test_mask_contour, dataset_test_contour = database.dataset_contours(dataset_train_mask, dataset_test_mask, dataset_train, dataset_test)
    # database.dataset_visualisation(dataset_train_mask_contour, labels_train, dataset_test_mask_contour, labels_test, number_classes, percent, '_mask_contour')
    # database.dataset_visualisation(dataset_train_contour, labels_train, dataset_test_contour, labels_test, number_classes, percent, '_contour')

    # compute HOG on dataset train original cropped
    width, height = helpers.get_dimensions_for_resize(dataset_train_contour, dataset_test_contour)
    train_features, train_images_hog = features.extract_features_hog(dataset_train_contour, width, height, False)
    test_features, test_images_hog = features.extract_features_hog(dataset_test_contour, width, height, False)

    # classification with SVM
    clf = classification.svm_classification(train_features, labels_train)

    # prediction
    classification.prediction(clf, test_features, labels_test)