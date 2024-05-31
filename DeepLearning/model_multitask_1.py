# Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import json
from glob import glob
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input


# Parameters
folder_path = r"/home/amy/kunskapskontroll_deep_learning/"
image_size = 48
batch_size = 32
learning_rate = 0.0001
epochs = 100
model_save_name = "model_multitask_1.h5"
plot_name = 'plot_model_multitask_1.jpg'
training_history_file_name='training_history_multitask_1.json'



gender_to_index = {0: 'male', 1: 'female'}
race_to_index = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others_hispanic_middle_eastern'}

filenames, ages, genders, races = [], [], [], []
count = 0
for filename in glob('/home/amy/kunskapskontroll_deep_learning/UTKFace/*.jpg'):
    try:
        age, gender, race = (int(v) for v in filename.split('UTKFace/')[1].split('_')[:3])
    except ValueError:
        count += 1
        print(filename)
    else:
        filenames.append(filename), ages.append(age), genders.append(gender), races.append(race)
        
if count:
    print(f'Found {count} not conforming filenames')
    
faces = pd.DataFrame({
    'filename': filenames,
    'age': ages,
    'gender': genders,
    'race': races
})
del filenames, ages, genders, races
#debugging
#print(faces.head())

onehot_races = to_categorical(faces['race'].values)
faces['onehot_races'] = onehot_races.tolist()
#debugging
#print(faces.head())



gen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    rotation_range=10,
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_set = gen.flow_from_dataframe(
    faces,
    y_col=['age', 'gender', 'onehot_races'],
    class_mode='multi_output',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='training'
)

val_set = gen.flow_from_dataframe(
    faces,
    y_col=['age', 'gender', 'onehot_races'],
    class_mode='multi_output',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    subset='validation'
)


base_model = VGG19(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
for layer in base_model.layers:
    layer.trainable=False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

age_prediction = Dense(1, name='age_prediction')(x)
gender_prediction = Dense(1, name='gender_prediction', activation='sigmoid')(x)
race_prediction = Dense(5, name='race_prediction', activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=[age_prediction, gender_prediction, race_prediction])  

opt = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=opt,
    loss=['mean_squared_error', 'binary_crossentropy','categorical_crossentropy'],
    metrics={'age_prediction': 'mean_absolute_error',
             'gender_prediction': 'accuracy',
            'race_prediction': 'accuracy'}
    )

checkpoint = ModelCheckpoint(
    folder_path + model_save_name,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
    )

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    verbose=1,
    restore_best_weights=True
)

reduce_learningrate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=4,
    verbose=1
)      

callbacks_list = [early_stopping, checkpoint, reduce_learningrate]
print("Callbacks_list created.")
print("model name:" + model_save_name)

training_history = model.fit(
    train_set,
    epochs=epochs,
    max_queue_size = 10,
    steps_per_epoch=train_set.n // batch_size,
    validation_data=val_set,
    validation_steps=val_set.n//batch_size,
    callbacks=callbacks_list
)



print("Training completed.")
model.save(folder_path + "training_results/" + model_save_name)

#print(training_history.history.keys())


def plot_training_history(history, save_path, plot_name):
    # Create a figure with three subplots
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


print("Training history plots saved.")

history_dict = {key: [float(value) for value in values] for key, values in training_history.history.items()}

with open(folder_path + training_history_file_name, 'w') as json_file:
    json.dump(history_dict, json_file)

print("Training history saved to JSON file." + training_history_file_name)

#print(training_history.history.keys())

