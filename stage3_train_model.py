# REQUIRED LIBRARIES

# Importing the function to compute metrics and visualizations
from stage4_metrics_function import compute_metrics_and_visualizations
# Saving the model history to pickle format
import pickle
# Importing the function to create a CNN model from the created model module
from stage2_create_model import create_cnn_model
# Working with numpy arrays
import numpy as np
# Splitting the dataset into training and testing sets
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
    # Load the numpy arrays containing the data and target
    data = np.load('data.npy')
    target = np.load('target.npy')
    # Shape of the images
    image_shape = data.shape[1:]
    print('\nImages shape:', image_shape)

    return data, target


# Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    # Define callbacks for model training and saving the best model
    checkpoint = ModelCheckpoint(
        # Save model weights with epoch number
        'model-{epoch:03d}.model',
        # Monitor validation loss
        monitor='val_loss',
        # Display verbose output
        verbose=1,
        # Save only the best model
        save_best_only=True,
        # Automatically determine the best model based on validation loss
        mode='auto',
        # Save the entire model, not just weights
        save_weights_only=False
    )

    # To stop training if no improvement in val_loss
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=10,  # Number of epochs with no improvement before stopping
        verbose=1,    # Display verbose output
        restore_best_weights=True  # Restore weights of the best perform model
    )

    # To control learning rate during training
    reduce_lr = ReduceLROnPlateau(
        # Monitor validation loss
        monitor='val_loss',
        # Factor by which the learning rate will be reduced
        factor=0.2,
        # Number of epochs with no improvement - learning rate will be reduced
        patience=5,
        # Minimum learning rate
        min_lr=1e-6
    )

    # To define the TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir='./logs',
        # The directory where TensorBoard will write its log files
        histogram_freq=1,  # Histograms will be computed and logged every epoch
        write_graph=True,  # Writing the computation graph to the log files
        write_images=True  # Writing the model weights as images
    )

    # Data augmentation parameters
    datagen = ImageDataGenerator(
        # The images can be rotated randomly between -20 and +20 degrees
        rotation_range=20,
        # The input images can be randomly shifted horizontally
        width_shift_range=0.2,
        # The input images can be randomly shifted vertically
        height_shift_range=0.2,
        # The range within which shearing transformations can be applied
        shear_range=0.2,
        # The range for randomly zooming into the input images
        zoom_range=0.2,
        # Randomly flip the input images horizontally
        horizontal_flip=True,
        # Strategy to fill in missing pixels after applying transformations
        fill_mode='nearest'
    )

    # Train the model with data augmentation
    history = model.fit(
        # Generate augmented batches of data using the data generator
        datagen.flow(X_train, y_train, batch_size=batch_size),
        # Define the number of steps (batches) to be processed in each epoch
        steps_per_epoch=len(X_train) / batch_size,
        # Specify the number of epochs for training
        epochs=50,
        # Provide validation data to monitor model performance during training
        validation_data=(X_val, y_val),
        # Specify the callbacks to be used during training
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback],
    )

    # Save the training history using pickle
    with open('trainHistoryDict.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return history


# Execute the main function if the script is run directly
if __name__ == "__main__":
    # Load data
    data, target = load_data()

    # This splits the entire dataset into training and testing sets:
    # Training Set: 80% of data; Test Set: 20% of data
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2,
                                                        random_state=42)

    # This splits the training set into a smaller training and validation sets:
    # Training Set: 70% of original data (which is 70% of 80% = 56%)
    # Validation Set: 30% of original training data (which is 30% of 80% = 24%)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.3,
                                                      random_state=42)

    # By performing these two splits, I achieve the following distribution:
    # Training Set: 70% of the original data (X_train, y_train)
    # Validation Set: 24% of the original data (X_val, y_val)
    # Test Set: 20% of the original data (X_test, y_test)

    # Create the CNN model
    model = create_cnn_model(X_train.shape[1:])

    # Train the model and get the training history
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Compute metrics and visualize
    compute_metrics_and_visualizations(model, X_test, y_test, history)
