import os
import sys
import tensorflow as tf
import numpy as np

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.models.cnn_model import CNNModel

# Register the custom CNNModel class
with tf.keras.utils.custom_object_scope({"CNNModel": CNNModel}):
    # Load the saved model
    model = tf.keras.models.load_model("cnn_model.h5")

# Load some test data (for example, the first image from the MNIST test set)
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Predict the label of the first test image
predictions = model.predict(test_images[:1])
predicted_label = np.argmax(predictions[0])
print(f"Predicted label: {predicted_label}")
print(f"True label: {test_labels[0]}")
