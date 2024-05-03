# REQUIRED LIBRARIES

import os
# Reading the file with model history
import pickle
import numpy as np
from stage3_train_model import load_data, train_model
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def load_history(file_path):
    with open(file_path, 'rb') as file_pi:
        history = pickle.load(file_pi)
    return history


history_file = 'trainHistoryDict.pkl'
history = load_history(history_file)

print(history.keys())  # Print the keys of the loaded dictionary


def main():
    # Check if the trained model and history file exist
    model_file = 'model-018.model'
    history_file = 'trainHistoryDict.pkl'

    if not os.path.exists(model_file):
        print(f'Error: Trained model file "{model_file}" not found.')
        return

    if not os.path.exists(history_file):
        print(f'Error: History file "{history_file}" not found.')
        return

    # Load data
    data, target = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # Load the trained model
    model = load_model(model_file)
    print(f'Pre-trained "{model_file}" loaded successfully.')

    # Load the training history
    history = load_history(history_file)

    # Plot loss if history is not None
    if history is not None:
        plt.plot(history['loss'], 'r', label='Training loss')
        plt.plot(history['val_loss'], label='Validation loss')
        plt.title(f'Training and Validation Loss: {model_file}', color='navy', size=14)
        plt.xlabel('Epochs', color='navy', size=12)
        plt.ylabel('Loss', color='navy', size=12)
        plt.legend()
        plt.show()

        # Plot accuracy
        plt.plot(history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy: {model_file}', color='navy', size=14)
        plt.xlabel('Epoch', color='navy', size=12)
        plt.ylabel('Accuracy', color='navy', size=12)
        plt.legend()
        plt.show()  # Accuracy of the model.
        print(model.evaluate(X_test, y_test))

        # Generate confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        # cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        # plt.imshow(cm, cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix', color='navy', size=14)
        # plt.colorbar()
        # plt.xlabel('Predicted labels', color='navy', size=12)
        # plt.ylabel('True labels', color='navy', size=12)
        # plt.show()

        # Generate classification report
        print("Classification Report:")
        print(classification_report(np.argmax(y_test, axis=1), y_pred))


if __name__ == "__main__":
    main()
