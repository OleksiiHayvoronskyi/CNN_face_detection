# REQUIRED LIBRARIES

from stage4_metrics_function import compute_metrics_and_visualizations
# Saving the file with model history to pickle format
import pickle
# Import 'create_cnn_model()' function from stage2_create_model.py
from stage2_create_model import create_cnn_model
# Work with numpy array
import numpy as np
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
# For saving the model during training based on certain conditions
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Controlling the learning rate
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Creating a TensorBoard callback
from tensorflow.keras.callbacks import TensorBoard
# For applying different transformations to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define batch size:
# For defining the number of samples that will be propagated through
# the neural network at each iteration
batch_size = 32


# Load the data
def load_data():
    data = np.load('data.npy')
    target = np.load('target.npy')
    # Shape of the images
    image_shape = data.shape[1:]
    print('\nImages shape:', image_shape)

    return data, target


# Train the model
def train_model(model, X_train, y_train):
    # Define callbacks for model training and saving the best model
    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.model',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto',
        save_weights_only=False
    )

    # To stop training if no improvement in val_loss
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    # To control learning rate during training
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        factor=0.2,        # Factor by which the learning rate will be reduced
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6          # Minimum learning rate
    )

    # To define the TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir='./logs',
        # The directory where TensorBoard will write its log files
        histogram_freq=1,  # The histograms will be computed and logged every epoch
        write_graph=True,  # Writing the computation graph to the log files
        write_images=True  # Writing the model weights as images
    )

    # Splitting the data into training and validation sets for data augmentation
    X_train, X_val, y_train, y_val = train_test_split(data, target,
                                                      test_size=0.3,
                                                      random_state=42)

    # Data augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20,   # The images can be rotated randomly between -20 and +20 degrees
        width_shift_range=0.2,  # The input images can be randomly shifted horizontally
        height_shift_range=0.2,  # The input images can be randomly shifted vertically
        shear_range=0.2,  # The range within which shearing transformations can be applied
        zoom_range=0.2,  # The range for randomly zooming into the input images
        horizontal_flip=True,  # Randomly flip the input images horizontally
        fill_mode='nearest'  # Strategy to fill in missing pixels after applying transformations
    )

    # Train the model with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback],
    )

    # Save the training history using pickle
    with open('trainHistoryDict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history


if __name__ == "__main__":
    # Load data
    data, target = load_data()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2,
                                                        random_state=42)

    # Create the CNN model
    model = create_cnn_model(X_train.shape[1:])

    # Train the model and get the training history
    history = train_model(model, X_train, y_train)

    # Compute metrics and visualize
    compute_metrics_and_visualizations(model, X_test, y_test, history)
