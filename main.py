import src.database as dataset

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
    dataset_train, labels_train, dataset_test, labels_test, number_classes = dataset.get_train_and_test_dataset(data_dir, percent, scale_factor)
    dataset.dataset_visualisation(dataset_train, labels_train, dataset_test, labels_test, number_classes, percent)