import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img



def conf_image_and_folder(image_size, folder_path):
    print(f"The image size is set to: {image_size} \nThe folder_path is set to: {folder_path}")
    return image_size, folder_path

def observe_images(expression, image_size, folder_path):
    """Available expressions are 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'"""
    plt.figure(figsize=(12,12))
    print(f'{expression} images')
    for i in range(1, 10):
        plt.subplot(3,3,i)
        img = load_img(folder_path +"/images_3_splits/train/" + expression + "/" + os.listdir(folder_path +"/images_3_splits/train/" + expression)[i],
                  target_size=(image_size, image_size))
        plt.imshow(img)
        print(os.listdir(folder_path +"/images_3_splits/train/" + expression)[i])

    return plt.show()

def train_val_preprocessing(folder_path, batch_size, color_mode, image_size):
    datagen_train = ImageDataGenerator(rescale=1.0/255.0)
    datagen_val = ImageDataGenerator(rescale=1.0/255.0)
    train_set = datagen_train.flow_from_directory(folder_path + "/RAF_DB_train",
                                             target_size=(image_size, image_size),
                                             color_mode=color_mode,
                                             batch_size=batch_size,
                                             class_mode="categorical",
                                             shuffle=True)
    val_set = datagen_val.flow_from_directory(folder_path + "/RAF_DB_val",
                                             target_size=(image_size, image_size),
                                             color_mode=color_mode,
                                             batch_size=batch_size,
                                             class_mode="categorical",
                                             shuffle=True)
    print("Train set ready")
    return train_set, val_set


def train_val_test_preprocessing(folder_path, batch_size, image_size, color_mode):
    datagen_train = ImageDataGenerator(rescale=1.0/255.0)
    datagen_val = ImageDataGenerator(rescale=1.0/255.0)
    datagen_test = ImageDataGenerator(rescale=1.0/255.0)


    train_set = datagen_train.flow_from_directory(folder_path+"/images_3_splits/train",
                                             target_size=(image_size, image_size),
                                             color_mode=color_mode,
                                             batch_size=batch_size,
                                             class_mode="categorical",
                                             shuffle=True)
    val_set = datagen_val.flow_from_directory(folder_path+"/images_3_splits/validation",
                                             target_size=(image_size, image_size),
                                             color_mode=color_mode,
                                             batch_size=batch_size,
                                             class_mode="categorical",
                                             shuffle=True)
    test_set = datagen_test.flow_from_directory(folder_path+"/images_3_splits/test",
                                             target_size=(image_size, image_size),
                                             color_mode=color_mode,
                                             batch_size=batch_size,
                                             class_mode="categorical",
                                             shuffle=True)
    print("Train, val, test sets ready")
    return train_set, val_set, test_set

def train_set_aug_and_preprocess(folder_path, batch_size, image_size, color_mode):
    datagen_train_augmented = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_train_set = datagen_train_augmented.flow_from_directory(
        folder_path+"/images_3_splits/train",
        target_size=(image_size, image_size),
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    print("Augmented train set ready")
    return augmented_train_set