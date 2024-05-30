# import functions
from tensorflow.keras.applications import ResNet50
from functions_data_preprocessing import train_val_test_preprocessing
from functions_model import build_model_1, create_callbacks_list, fit_model
from functions_result import save_training_history, plot_training_history, test_model, print_best_training_result

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model,Sequential,load_model
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


folder_path = "/home/amy/kunskapskontroll_deep_learning"
batch_size = 32
image_size = 48
learning_rate = 0.0001
model_name = "model_4"
model_save_name = "model_4.h5"
epochs = 50
plot_name = 'plot_model_4.jpg'
training_history_file_name='training_history_4.json'

#preprocessing
train_set, val_set, test_set = train_val_test_preprocessing(
    folder_path=folder_path, 
    batch_size=batch_size,
    image_size=image_size,
    color_mode="rgb")


# load the VGG16 network and initialize the label encoder
Resnet50v2_base_model = ResNet50(weights="imagenet", include_top=False, input_shape= (image_size, image_size, 3))

#freeze all layers
for layer in Resnet50v2_base_model.layers:
    layer.trainable = False

#create a new top layer for the model
x = Resnet50v2_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

predictions = Dense(7, activation='softmax')(x)  # Output layer for 7 classes

model = Model(inputs=Resnet50v2_base_model.input, outputs=predictions)

opt = Adam(learning_rate=learning_rate)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks_list = create_callbacks_list(folder_path=folder_path, model_name=model_name)

training_history = fit_model(
    model=model, 
    training_dataset=train_set, 
    validation_dataset=val_set, 
    epochs=epochs, 
    callbacks_list=callbacks_list, 
    batch_size=batch_size, 
    folder_path=folder_path,
    model_name=model_name)


# Training result documentation
plot_training_history(training_history=training_history, plot_name=plot_name)
save_training_history(training_history=training_history, training_history_file_name=training_history_file_name)
