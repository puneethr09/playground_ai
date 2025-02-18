import os
import sys
import numpy as np
import tensorflow as tf

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.data_loader import load_processed_data
from src.models.cnn_model import CNNModel


def train_model():
    # Load the processed data
    (train_data, train_labels), (test_data, test_labels) = load_processed_data()

    # Print the shapes of the data to debug
    print(f"train_data shape: {train_data.shape}")
    print(f"train_labels shape: {train_labels.shape}")

    # Initialize the CNN model
    input_shape = (train_data.shape[1], train_data.shape[2], train_data.shape[3])
    num_classes = len(np.unique(train_labels))
    model = CNNModel(input_shape=input_shape, num_classes=num_classes)

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        train_data,
        train_labels,
        epochs=10,
        batch_size=32,
        validation_data=(test_data, test_labels),
    )

    # Save the model
    model.save("cnn_model.h5")
    print("Model saved to cnn_model.h5")


if __name__ == "__main__":
    train_model()
