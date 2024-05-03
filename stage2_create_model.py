# REQUIRED LIBRARIES

# For interacting with the operating system
import os
# Turn off the use of custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Work with numpy array
import numpy as np
from tensorflow.keras.models import Sequential
# For constructing the neural network
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
# L2 regularization
from keras.regularizers import l2
# For building Convolutional Layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# For optimizing the neural network: Adam
from tensorflow.keras.optimizers import Adam
# For creating visualizations
from matplotlib import pyplot as plt
# Creating a visual representation of the model's architecture
from tensorflow.keras.utils import plot_model


# Load the saved numpy arrays
def load_data():
    # Loading the saved numpy arrays
    # The data for the neural network
    data = np.load('data.npy')
    # This array contains the corresponding target labels for the input data
    target = np.load('target.npy')
    return data, target


# Initialize a sequential keras model
def create_cnn_model(input_shape):
    # Initializing a sequential keras model (a linear stack of layers)
    model = Sequential()

    # INPUT LAYER:
    # The first convolution CNN layer with 200 filters, each of size (3, 3),
    # where 1 (into input_shape) represents the number of color channels (gray)
    model.add(Conv2D(200, (3, 3), input_shape=input_shape))
    # HIDDEN LAYERS:
    # Adding an activation layer.
    model.add(Activation('relu'))
    # Adding a max-pooling layer.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # HIDDEN LAYERS:
    # The second convolution (CNN) layer:
    model.add(Conv2D(100, (3, 3)))
    # Adding an activation layer.
    model.add(Activation('relu'))
    # Adding a max-pooling layer.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # HIDDEN LAYERS:
    # This layer flattens the 3D output from the previous layer
    # into a 1D vector, repairing it for the dense layers
    model.add(Flatten())
    # Adding a dropout layer with a dropout rate of 0.5 to prevent overfitting
    model.add(Dropout(0.5))

    # HIDDEN LAYER:
    # Single Dense layer with 100 neurons and L2 regularization
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))

    # OUTPUT LAYER:
    # The Final layer with five output neurons.
    # Neurons represent my faces and my friends' faces
    model.add(Dense(5, activation='softmax'))

    # Compiling the model with specified settings
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Printing the model summary.
    print('--- Created Convolutional Neural Network ---')
    model.summary()

    return model


# Visualize the model architecture
def visualize_model(model, filename='model_plot.png'):
    # Visualize the model architecture
    plot_model(model, to_file=filename,
               show_shapes=True,
               show_layer_names=True)

    # Display the model plot using matplotlib
    img = plt.imread(filename)
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(img)
    plt.show()
