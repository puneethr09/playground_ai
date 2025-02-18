import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_raw_data(data_dir):
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    data_frames = [pd.read_csv(os.path.join(data_dir, f)) for f in data_files]
    return pd.concat(data_frames, ignore_index=True)


def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna()  # Remove missing values
    # Add more preprocessing steps as needed
    return df


def split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df, test_df


def save_processed_data(train_df, test_df, output_dir):
    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)


def load_data(file_path):
    """Load raw data from a specified file path."""
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """Clean the data by handling missing values and duplicates."""
    data = data.dropna()
    data = data.drop_duplicates()
    return data


def load_processed_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape data to fit the model
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    return (train_images, train_labels), (test_images, test_labels)
