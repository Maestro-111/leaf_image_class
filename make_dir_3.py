import os
import random
import shutil

# Define the source directory
leafs_dir = "Leafs_Structured"

# Define the destination directory
dataset_dir = "Dataset"

test_percentage = 0.1
validation_percentage = 0.15

# Loop through each leaf type folder
for leaf_type in os.listdir(leafs_dir):

    leaf_type_dir = os.path.join(leafs_dir, leaf_type)  # leadf/type_sick

    image_files = os.listdir(leaf_type_dir)

    train_path = f"{dataset_dir}/train/{leaf_type}"
    test_path = f"{dataset_dir}/test/{leaf_type}"
    validation_path = f"{dataset_dir}/validation/{leaf_type}"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    num_images = len(image_files)
    num_test = int(num_images * test_percentage)
    num_validation = int(num_images * validation_percentage)
    num_train = num_images - (num_test + num_validation)

    # Randomly select and move images to the test set
    random.shuffle(image_files)

    test_files = image_files[:num_test]
    validation_files = image_files[num_test:num_test + num_validation]
    train_files = image_files[num_test + num_validation:]

    for file in test_files:
        shutil.copy(os.path.join(leaf_type_dir, file), os.path.join(test_path, file))

    for file in validation_files:
        shutil.copy(os.path.join(leaf_type_dir, file), os.path.join(validation_path, file))

    for file in train_files:
        shutil.copy(os.path.join(leaf_type_dir, file), os.path.join(train_path, file))