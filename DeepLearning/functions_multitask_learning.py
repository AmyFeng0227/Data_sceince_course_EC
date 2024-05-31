import os
import numpy as np
import cv2

# TensorFlow and Keras imports
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model



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

