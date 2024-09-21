# import functions
from functions_data_preprocessing import train_val_preprocessing
from functions_model import build_model_1, create_callbacks_list, fit_model
from functions_result import save_training_history, plot_training_history

# Configuration and Parameters
image_size = 48
folder_path = "/home/amy/kunskapskontroll_deep_learning"
batch_size = 32
learning_rate = 0.0001
model_name = "model_1_3"
model_save_name = "model_1_3.h5"
epochs = 100
plot_name = 'plot_model_1_3.jpg'
training_history_file_name='training_history_1_3.json'
color_input = 3


# Data Preprocessing
train_set, val_set = train_val_preprocessing(
    folder_path=folder_path, 
    batch_size=batch_size, 
    color_mode="rgb", 
    image_size=image_size)


# Model creation and training
model = build_model_1(image_size=image_size, learning_rate=learning_rate, color_input=color_input)

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
