import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
# TensorFlow and Keras imports
from keras.models import load_model


def plot_training_history(training_history, plot_name):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.suptitle('Optimizer: Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(training_history.history['loss'], label='Training loss')
    plt.plot(training_history.history['val_loss'], label='Validation loss')
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(training_history.history['accuracy'], label='Training accuracy')
    plt.plot(training_history.history['val_accuracy'], label='Validation accuracy')
    plt.legend(loc='lower right')

    # Save the plot
    training_results_folder_path = r'/home/amy/kunskapskontroll_deep_learning/training_results/'
    plot_path = os.path.join(training_results_folder_path, plot_name)
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
def test_model(model, test_set, test_result_file_name):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_set)
    
    # Predict the labels
    predictions = model.predict(test_set)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Get the true labels
    true_labels = test_set.classes
    
    # Generate classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=test_set.class_indices.keys(), output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Save the results to a file
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist()  # Convert to list for JSON serialization
    }
    
    results_file_path = os.path.join('/home/amy/kunskapskontroll_deep_learning/training_results/', test_result_file_name)
    with open(results_file_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Test results saved to {results_file_path}")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print("Classification Report:")
    print(json.dumps(class_report, indent=2))
    print("Confusion Matrix:")
    print(conf_matrix)
    return results

def save_training_history(training_history, training_history_file_name):
    training_results_folder_path = r'/home/amy/kunskapskontroll_deep_learning/training_results/'
    history_path = os.path.join(training_results_folder_path, training_history_file_name)
    history_dict = {key: [float(val) for val in values] for key, values in training_history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f)
    return f"History saved to {history_path}"

def load_training_history(training_history_file_name):
    training_results_folder_path = r'/home/amy/kunskapskontroll_deep_learning/training_results/'
    history_path = os.path.join(training_results_folder_path, training_history_file_name)
    with open(history_path, 'r') as f:
        training_history = json.load(f)
    return training_history

def print_best_training_result(training_history, training_history_file_name):
    best_epoch = max(range(len(training_history['val_accuracy'])), key=lambda i: training_history['val_accuracy'][i])
    #print("The best training result for" + training_history_file_name + "is:")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Training loss: {training_history['loss'][best_epoch]:.3f}")
    print(f"Training accuracy: {training_history['accuracy'][best_epoch]:.3f}")
    print(f"Validation loss: {training_history['val_loss'][best_epoch]:.3f}")
    print(f"Validation accuracy: {training_history['val_accuracy'][best_epoch]:.3f}")


def predict_with_model(model_path, image_path, image_size):
    model = load_model(model_path)
    class_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

    #image preprocessing
    img = cv2.imread(image_path)  
    img = cv2.resize(img, (image_size, image_size))  
    img = img / 255.0  
    img_array = np.expand_dims(img, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0) 

    #prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    for label, prob in zip(class_labels, predictions[0]):
        print(f"{label}: {prob:.3f}")

    print(f"Predicted label: {predicted_class_label}")

def summarise_result(training_history_file_name, image_path):
    training_history = load_training_history(training_history_file_name=training_history_file_name)
    print_best_training_result(training_history, training_history_file_name=training_history_file_name)
    image = Image.open(image_path)
    plt.figure(figsize=(14, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()