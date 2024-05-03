# CNN Face Detection Project

## Overview

This project utilizes Convolutional Neural Networks (CNNs) to detect faces in images.  
It is designed to work with a dataset of 132 images containing various faces.  
The trained model can then be used for real-time face detection using a camera.


## Files

- 'stage_1_preprocessing.py: Script for preprocessing the dataset.  
- stage2_create_model.py: Script for creating the CNN model architecture.  
- stage2.1_show_model_main: Implement model display functionality.  
- stage3_train_model.py: Script for training the CNN model.  
- stage3_1_metrics_for_certain_model.py: Script for evaluating metrics for a certain model.  
- stage4_metrics_function.py: Script containing functions for computing metrics and visualizations.  
- stage5_final_stage_face_detect.py: Script for performing real-time face detection.</br>


## Dataset

The dataset consists of 132 images of my friends' faces.  
These images are used for training and evaluating the CNN model for face detection.  
Each image contains one or more faces.


## Requirements

- Python 3.x
- IDE: PyCharm
- TensorFlow 2.x
- OpenCV (for real-time face detection with a camera)
- Other dependencies as specified in requirements.txt


## Usage

- Clone this repository: git clone https://github.com/OleksiiHayvoronskyi/CNN_face_detection  
- Install the required dependencie: pip install -r requirements.txt  
- Train the CNN model: python stage3_train_model.py  
- Once the model is trained, you can perform face detection using the camera: python stage5_final_stage_face_detect.py


## Contributors

**Oleksii Haivoronskyi**




