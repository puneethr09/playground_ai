import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.models.cnn_model import CNNModel

# Register the custom CNNModel class and load the saved model
with tf.keras.utils.custom_object_scope({"CNNModel": CNNModel}):
    model = tf.keras.models.load_model("cnn_model.h5")

# Open the webcam (0 is usually the built-in webcam on a Mac)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(
    1, cv2.CAP_AVFOUNDATION
)  # Open the iPhone camera (index 1 for iPhone)

if not cap.isOpened():
    print("Unable to access the webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28 as expected by the model
    resized = cv2.resize(gray, (28, 28))
    # Normalize pixel values
    normalized = resized / 255.0
    # Reshape to (1, 28, 28, 1)
    input_img = normalized.reshape(1, 28, 28, 1)

    # Predict digit and get the label
    predictions = model.predict(input_img)
    predicted_label = np.argmax(predictions[0])

    # Display prediction on the frame with black text, larger font scale, and increased thickness
    cv2.putText(
        frame,
        f"Predicted: {predicted_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
    )

    cv2.imshow("Live Inference", frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
