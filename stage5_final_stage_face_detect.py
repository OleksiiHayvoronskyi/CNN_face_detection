from keras.models import load_model
import cv2
import numpy as np


# Load the pre-trained face recognition model
model = load_model('model-018.model')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Checking if a camera can be open.
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Import cascade file for facial recognition.
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Import cascade file for smile recognition.
smile_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Dictionary to map labels to your friends' names
labels_dict = {0: 'Iryna', 1: 'It\'s_Me', 2: 'Serhiy',
               3: 'Tymur', 4: 'Yaroslav', 5: 'Other people'}
# Index corresponding to your face
your_face_label = 1

# Color dictionary for bounding box colors
color_dict = {1: (0, 255, 0)}  # Green color assigned for label 1 (It's_Me)

# for label in range(0, 6):      # Iterating over labels 0 to 5
#     if label != 1:  # Check if the label is not 1
#         color_dict[label] = (255, 0, 0)  # Assign blue color for labels other than 1

# Iterate over labels 0 to 6
for label in range(0, 6):
    # Check if the label is not 1 (your face)
    if label != 1:
        # Assign blue color for labels other than 1
        if label == 5:
            color_dict[label] = (0, 0, 255)  # Red color for "Other people"
        else:
            color_dict[label] = (255, 0, 0)  # Blue color for other labels

rect_size = 4

# Define the codec and create VideoWriter object.
# The output is stored in 'output.avi' file.
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

# Main loop for capturing video and performing face detection
while True:
    # Read a frame from the webcam
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print('[INFO] detecting faces...')

    # Detect faces using Haar Cascade Classifier
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))

        # Make predictions using the face mask detection model
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box around the face and display label
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], rect_size)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[label], 2)

        # Detect smiles within the face region
        smiles = smile_clsfr.detectMultiScale(face_img, scaleFactor=1.8,
                                              minNeighbors=20)

        # Loop through detected smiles
        # for (sx, sy, sw, sh) in smiles:
        #     # Draw bounding box around the smile
        #     cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh),
        #                   (0, 255, 255), 2)

        # Loop through detected smiles
        for (sx, sy, sw, sh) in smiles:
            # Calculate the center of the smile region
            center = (x + sx + sw // 2, y + sy + sh // 2)

            # Calculate the axes lengths of the ellipse (circle in this case)
            axes_length = (sw // 2, sh // 2)

            # Draw the ellipse (circle) around the smile
            cv2.ellipse(img, center, axes_length, 0, 0, 360, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('LIVE', img)

    # Write the frame to the output video file
    out.write(img)

    # Check for the 'Esc' key to exit he loop
    key = cv2.waitKey(1)
    if key == 27:  # ASCII value for Esc key / Ctrl+C
        break

# Release the webcam, video writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
