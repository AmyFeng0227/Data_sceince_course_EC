from functions_result import predict_with_model
from functions_data_preprocessing import train_val_test_preprocessing
from functions_model import build_model_1, create_callbacks_list, fit_model
from functions_result import save_training_history, plot_training_history, load_training_history, print_best_training_result


print("training_history_1_3_backup")
training_history = load_training_history(
    training_history_file_name="training_history_1_3_backup.json")

print_best_training_result(training_history=training_history, training_history_file_name="training_history_1_3_backup.json")

print("training_history_1_3")

training_history = load_training_history(
    training_history_file_name="training_history_1_3.json")

print_best_training_result(training_history=training_history, training_history_file_name="training_history_1_3.json")


print("training_history_2_1")

training_history = load_training_history(
    training_history_file_name="training_history_2_1.json")

print_best_training_result(training_history=training_history, training_history_file_name="training_history_2_1.json")



model_path_1 = r"/home/amy/kunskapskontroll_deep_learning/model_1_3_checkpoint.keras"
model_path_2 = r"/home/amy/kunskapskontroll_deep_learning/model_1_3.h5"
model_path_3 = r'/home/amy/kunskapskontroll_deep_learning/model_2_1_checkpoint.keras'
image_path = r"/home/amy/kunskapskontroll_deep_learning/face pictures/neutral.jpg"
image_size = 48


predict_with_model(model_path=model_path_1, image_path=image_path, image_size=image_size)
predict_with_model(model_path=model_path_2, image_path=image_path, image_size=image_size)
predict_with_model(model_path=model_path_3, image_path=image_path, image_size=image_size)