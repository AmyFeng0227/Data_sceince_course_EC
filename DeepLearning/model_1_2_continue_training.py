from functions_data_preprocessing import train_val_preprocessing, conf_image_and_folder, train_val_test_preprocessing, train_set_aug_and_preprocess
from functions_model import build_model_1, create_callbacks_list, fit_model
from functions_result import save_training_history, plot_training_history, test_model, print_best_training_result
from tensorflow.keras.models import load_model

# Configuration and Parameters
image_size = 48
folder_path = "/home/amy/kunskapskontroll_deep_learning/"
batch_size = 32
learning_rate = 0.0001
model_name = "model_1_2_add"
epochs = 100
plot_name = 'plot_model_1_2_add.jpg'
training_history_file_name='training_history_1_2_add.json'

_, val_set, test_set = train_val_test_preprocessing(
    folder_path=folder_path, 
    batch_size=batch_size, 
    color_mode="grayscale", 
    image_size=image_size)

train_set = train_set_aug_and_preprocess(folder_path=folder_path, batch_size=batch_size, color_mode="grayscale",image_size=image_size)


model = load_model(folder_path + "training_results/model_1_2.h5")

callbacks_list = create_callbacks_list(folder_path=folder_path, model_name=model_name)

additional_history = fit_model(
    model=model,
    training_dataset=train_set,
    validation_dataset=val_set,
    epochs=epochs,
    callbacks_list=callbacks_list,
    batch_size=batch_size,
    model_name=model_name,
    folder_path=folder_path
)

# Training result documentation
plot_training_history(training_history=additional_history, plot_name=plot_name)
save_training_history(training_history=additional_history, training_history_file_name=training_history_file_name)