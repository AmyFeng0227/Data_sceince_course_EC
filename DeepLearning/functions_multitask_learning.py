import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
# TensorFlow and Keras imports
import tensorflow as tf






from functions_result import load_training_history

def load_utkface_data(image_folder_path, image_size):
    images = []
    ages = []
    genders = []
    races = []

    for filename in os.listdir(image_folder_path):
        if filename.endswith(".jpg"):
            try:
                # Split the filename to extract the age, gender, and race
                parts = filename.split("_")
                if len(parts) < 4:
                    print(f"Warning: Skipping file with unexpected filename format: {filename}")
                    continue

                age = parts[0]
                gender = parts[1]
                race = parts[2]
                img_path = os.path.join(image_folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: {img_path} could not be loaded.")
                    continue
                img = cv2.resize(img, (image_size, image_size))
                img = img / 255.0

                images.append(img)
                ages.append(int(age))
                genders.append(int(gender))
                races.append(int(race))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)
    races = np.array(races)
    print("images, ages, genders, races created.")
    return images, ages, genders, races


def build_multi_task_model(image_size, learning_rate, base_model):
    model = base_model(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output for age
    age_output = Dense(1, activation='linear', name='age_output')(x)
    
    # Output for gender
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    
    # Output for race
    race_output = Dense(5, activation='softmax', name='race_output')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output, race_output])
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss={'age_output': 'mean_squared_error',
                        'gender_output': 'categorical_crossentropy',
                        'race_output': 'categorical_crossentropy'},
                  metrics={'age_output': 'mae',
                           'gender_output': 'accuracy',
                           'race_output': 'accuracy'})
    return model

def plot_training_history(training_history_file_name, save_path, plot_name):
    # Create a figure with three subplots
    history = load_training_history(training_history_file_name=training_history_file_name)
    history = history.history 
    fig, axes = plt.subplots(6, 1, figsize=(12, 18))
    
    # Plot training & validation loss values for age prediction
    axes[0].plot(history['age_prediction_loss'], label='Train Loss')
    axes[0].plot(history['val_age_prediction_loss'], label='Validation Loss')
    axes[0].set_title('Age Prediction Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Plot training & validation MAE values for age prediction
    axes[1].plot(history['age_prediction_mean_absolute_error'], label='Train MAE')
    axes[1].plot(history['val_age_prediction_mean_absolute_error'], label='Validation MAE')
    axes[1].set_title('Age Prediction Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # Plot training & validation loss values for gender prediction
    axes[2].plot(history['gender_prediction_loss'], label='Train Loss')
    axes[2].plot(history['val_gender_prediction_loss'], label='Validation Loss')
    axes[2].set_title('Gender Prediction Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    # Plot training & validation accuracy values for gender prediction
    axes[3].plot(history['gender_prediction_accuracy'], label='Train Accuracy')
    axes[3].plot(history['val_gender_prediction_accuracy'], label='Validation Accuracy')
    axes[3].set_title('Gender Prediction Accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend(loc='upper left')
    axes[3].grid(True)

    # Plot training & validation loss values for race prediction
    axes[4].plot(history['race_prediction_loss'], label='Train Loss')
    axes[4].plot(history['val_race_prediction_loss'], label='Validation Loss')
    axes[4].set_title('Race Prediction Loss')
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('Loss')
    axes[4].legend(loc='upper right')
    axes[4].grid(True)

    # Plot training & validation accuracy values for race prediction
    axes[5].plot(history['race_prediction_accuracy'], label='Train Accuracy')
    axes[5].plot(history['val_race_prediction_accuracy'], label='Validation Accuracy')
    axes[5].set_title('Race Prediction Accuracy')
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Accuracy')
    axes[5].legend(loc='upper left')
    axes[5].grid(True)
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path + plot_name)
    plt.close()
