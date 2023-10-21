from tensorflow import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from preprocess_image import construct_data
from preprocess_image import code_dict
from preprocess_image import size_i
from preprocess_image import classes
from preprocess_image import create_datasets
from sklearn.model_selection import train_test_split
import os
from skimage import io, color
from matplotlib import cm
from keras.layers import Dropout
from keras.layers import Rescaling
from keras.regularizers import l2
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import tensorflow as tf

import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_confusion_matrix(cm, percent=False, categories=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    if percent:
        cm = np.round(100 * cm / cm.sum(), 2)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    if categories is not None:
        tick_marks = np.arange(len(categories))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(categories, rotation=45)
        ax.set_yticklabels(categories)

    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='w')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


class neural_net_mixin:

    """
    mutual methods for neural nets
    """

    def train_test(self,train,test,optimizer="adam",loss = "categorical_crossentropy", metrics=["accuracy"],
                   batch_size=20, epochs=50,validation_split=0.15):

        self.model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        h = self.model.fit(train, batch_size=batch_size, epochs=epochs)

        print("Done Training")

        self.model.evaluate(test)

        return h


    def predict(self,X,y,batch=20):

        reversed_dict = {value: key for key, value in code_dict.items()}
        pred = self.model.predict(X)

        print(pred)

        y_spare = []

        for l in y:
            index = np.where(l == 1)[0]
            if index <= 6:
                y_spare.append(["Sick",reversed_dict[index]])
            else:
                y_spare.append(["Healthy", reversed_dict[index-7]])

        y = y_spare

        pred = np.argmax(pred, axis=1)

        pred_spare = []

        for index in pred:
            if index <= 6:
                pred_spare.append(["Sick",reversed_dict[index]])
            else:
                pred_spare.append(["Healthy", reversed_dict[index-7]])

        pred = pred_spare


        print(y)
        print(pred)
        print(confusion_matrix(y, pred))

        batch = min(batch,len(pred))
        for i in range(batch):
            leaf_type_actual = y[i][1]
            left_type_predicted = pred[i][1]
            leaf_class_actual = y[i][0]
            leaf_class_predicted = pred[i][0]
            print(f"Class: {leaf_type_actual+' '+leaf_class_actual}, Answer: {left_type_predicted+' '+leaf_class_predicted}")

    @staticmethod
    def plot_learning_curves(history):
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_training_hist(hist, model_name: str, accuracy_colors, loss_colors):
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(hist.history['accuracy'], color=accuracy_colors[0])
        ax1.plot(hist.history['val_accuracy'], color=accuracy_colors[1])
        ax1.legend(['train acc', 'validation acc'], loc='upper left')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')

        ax2 = ax1.twinx()
        ax2.plot(hist.history['loss'], color=loss_colors[0])
        ax2.plot(hist.history['val_loss'], color=loss_colors[1])
        ax2.legend(['train loss', 'validation loss'], loc='upper right')
        ax2.set_ylabel('loss')

        plt.title(f'{model_name} training accuracy and loss per epoch')
        plt.show()

    @staticmethod
    def plot_cm(labels, predictions, categories):
        cm = confusion_matrix(labels, predictions)
        make_confusion_matrix(cm, percent=False, categories=categories)




class neural_nertwork_Full(neural_net_mixin):

    """
    Simple neural network, which consists of SIZE (input param to the class) input neurons, 128 hidden and 2 out neurons. 2 Because we have 2 classes
    """

    def __init__(self,size,num_classes):

        self.model = keras.Sequential([
            Flatten(input_shape=(size, size, 1)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        print(self.model.summary())

class CNN_for_leafs(neural_net_mixin):

    """
    CNN neural net
    """

    def __init__(self,size,num_classes):
        self.model = keras.Sequential([
            Rescaling(1. / 255),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Conv2D(32, 3, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes)
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])


    def train(self, train, validation,batch_size=20, epochs=10):

        h = self.model.fit(train, validation_data=validation, batch_size=batch_size, epochs=epochs)
        print("Done Training")
        return h

    def evaluate_model(self,test_dataset,class_names):
        test_loss, test_acc = self.model.evaluate(test_dataset)
        print(f'test accuracy : {test_acc}')
        print(f'test loss : {test_loss}')

        # Confusion matrix
        y_true = np.array([])
        y_pred = np.array([])
        for X, y in test_dataset:
            y_part = np.argmax(self.model.predict(X), axis=1).flatten()
            y_pred = np.concatenate([y_pred, y_part])
            y_true = np.concatenate([y_true, y])

            y_true_batch_names = [class_names[int(y)] for y in y]
            y_pred_batch_names = [class_names[int(y)] for y in y_part]

            # Display actual and predicted class names for the current batch
            for i in range(len(y_true_batch_names)):
                actual_name = y_true_batch_names[i]
                predicted_name = y_pred_batch_names[i]
                print(f"Actual: {actual_name}, Predicted: {predicted_name}")
                print()


        test_f1 = f1_score(y_true, y_pred, average="macro")
        print(f'test f1-score : {test_f1}')

        self.plot_cm(y_true, y_pred, class_names)
        plt.show()



a = time.time()

train_dataset, validation_dataset, test_dataset, class_names, num_classes = create_datasets("C:/leaf_class/Dataset/train",
                                                                                            "C:/leaf_class/Dataset/validation",
                                                                                            "C:/leaf_class/Dataset/test")
print(class_names)
print(num_classes)


CNN_net = CNN_for_leafs(224,num_classes)
history = CNN_net.train(train_dataset,validation_dataset,epochs=10)
CNN_net.plot_training_hist(history, '3-layers CNN', ['red', 'orange'], ['blue', 'green'])
CNN_net.evaluate_model(test_dataset,class_names)



b = time.time()

print()
print(round((b-a)/60,2))
