import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, color, feature
from skimage import exposure
from skimage.transform import resize
from keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
import pickle



code_dict = {"Alstonia Scholaris (P2)":0, "Arjun (P1)":1,"Chinar (P11)":2,"Gauva (P3)":3,"Jamun (P5)":4,
             "Pomegranate (P9)":5,"Pongamia Pinnata (P7)":6}


class NoImageException(Exception):
    def __init__(self, message="This is a custom exception"):
        self.message = message
        super().__init__(self.message)


class construct_data:

    def __init__(self):
        self.hog_feature_vectors = []
        self.lables = []
        self.images = []

    def load_images(self,path,vis=False,batch_images=1,batch_types=50,resize_p=2000):
        file_list = os.listdir(path)
        initial_batch_types = batch_types
        for image_folder in sorted(file_list):

            if batch_images == 0:
                break

            image_label = code_dict[image_folder]

            for leaf_sick in os.listdir(f"{path}/{image_folder}"):  # label

                label = 1 if leaf_sick == "healthy" else -1


                for image_path in os.listdir(f"{path}/{image_folder}/{leaf_sick}"):

                    if batch_types == 0:
                        batch_types = initial_batch_types
                        break

                    image = imread(f"{path}/{image_folder}/{leaf_sick}/{image_path}")
                    image_size = np.shape(image)
                    image_width, image_height = image_size[1], image_size[0]

                    resize_p = min(resize_p,image_width,image_height)

                    print(F"Label is {label}, path is {image_folder}/{leaf_sick}/{image_path}")

                    #image = color.rgb2gray(resize(image, (resize_p, resize_p)))

                    image = resize(image, (resize_p, resize_p))

                    if label == -1:
                        self.lables.append(image_label)
                    else:
                        self.lables.append(image_label+7)

                    self.images.append(image)

                    print(f"Number of Images: {len(self.images)}")
                    if vis:
                        plt.imshow(image)  # Display the grayscale image
                        plt.axis('off')
                        plt.show()
                    batch_types -= 1
            batch_images -= 1

    def extract_hog_features(self,vis=False):

        if len(self.images) == 0:
            raise NoImageException("No Images!")

        for image in self.images:
            features, hog_image = hog(image,block_norm='L2-Hys', visualize=True)

            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

            self.hog_feature_vectors.append(features)

            if vis:
                plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)  # Display the grayscale image
                plt.axis('off')
                plt.show()

def create_datasets(directory_train, directory_valid, directory_test, batch_size=32, img_height=224, img_width=224, random_seed=42):

    train_dataset = image_dataset_from_directory(
        directory_train,
        seed=random_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    validation_dataset = image_dataset_from_directory(
        directory_valid,
        seed=random_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory_test,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    train_dataset = train_dataset.unbatch()
    validation_dataset = validation_dataset.unbatch()
    test_dataset = test_dataset.unbatch()

    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True)
    validation_dataset = validation_dataset.batch(batch_size=batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=True)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes

"""
if __name__ == "__main__":
    c = construct_data()
    c.load_images("Leafs",vis=False, batch_images=7,batch_types= 100,resize_p=size_i) # 3,50
    images = np.array(c.images)
    labels = np.array(c.lables)

    file1 = open("out1.bin", "wb")
    file2 = open("out2.bin", "wb")
    pickle.dump(images, file1)
    pickle.dump(labels, file2)
"""
classes = 7*2
size_i = 224

