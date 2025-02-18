import os
import sys
import numpy as np
import pandas as pd

# Add the project root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.utils.data_loader import load_data, clean_data


def preprocess_data(raw_data_path, output_data_path, output_labels_path):
    # Load raw data
    raw_data = load_data(raw_data_path)

    # Clean data
    cleaned_data = clean_data(raw_data)

    # Assuming the last column is the label
    labels = cleaned_data.iloc[:, -1].values
    data = cleaned_data.iloc[:, :-1].values

    # Generate synthetic data with shape (samples, 28, 28, 1)
    num_samples = data.shape[0]
    data = np.random.rand(num_samples, 28, 28, 1)

    # Save processed data as .npy files
    np.save(output_data_path, data)
    np.save(output_labels_path, labels)


if __name__ == "__main__":
    raw_data_path = "data/raw/raw_data.csv"  # Replace with your raw data file path
    output_data_path = "data/processed/train_data.npy"
    output_labels_path = "data/processed/train_labels.npy"

    preprocess_data(raw_data_path, output_data_path, output_labels_path)
    print("Data preprocessing complete. Processed data saved to data/processed/")
