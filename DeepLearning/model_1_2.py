# import functions
from functions_data_preprocessing import train_val_test_preprocessing
from functions_model import build_model_1, create_callbacks_list, fit_model
from functions_result import save_training_history, plot_training_historys

# Configuration and Parameters
image_size = 48
folder_path = "/home/amy/kunskapskontroll_deep_learning"
batch_size = 32
learning_rate = 0.0001
model_name = "model_1_2"
model_save_name = "model_1_2.h5"
epochs = 100
plot_name = 'plot_model_1_2.jpg'
training_history_file_name='training_history_1_2.json'


# Data Preprocessing
train_set, val_set, test_set = train_val_test_preprocessing(
    folder_path=folder_path, 
    batch_size=batch_size, 
    color_mode="grayscale", 
    model_type="own_CNN")


# Model creation and training
model = build_model_1(image_size=image_size, learning_rate=learning_rate)

callbacks_list = create_callbacks_list(folder_path=folder_path, model_name=model_name)

training_history = fit_model(
    model=model, 
    training_dataset=train_set, 
    validation_dataset=val_set, 
    epochs=epochs, 
    callbacks_list=callbacks_list, 
    batch_size=batch_size, 
    model_save_name= model_save_name, 
    folder_path=folder_path)

# Training result documentation
plot_training_history(training_history=training_history, plot_name=plot_name)
save_training_history(training_history=training_history, training_history_file_name=training_history_file_name)

