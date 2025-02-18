import pandas as pd


def load_data(file_path):
    """Load raw data from a specified file path."""
    data = pd.read_csv(file_path)
    return data


def clean_data(data):
    """Clean the data by handling missing values and duplicates."""
    data = data.dropna()
    data = data.drop_duplicates()
    return data


def process_data(file_path, output_path="data/processed/processed_data.csv"):
    """Process the raw data and save cleaned data to a CSV file."""
    raw_data = load_data(file_path)
    processed_data = clean_data(raw_data)
    processed_data.to_csv(output_path, index=False)  # Save to CSV
    return processed_data


if __name__ == "__main__":
    # Example usage:
    file_path = "data/raw/raw_data.csv"  # Replace with your raw data file path
    processed_data = process_data(file_path)
    print("Processed data saved to data/processed/processed_data.csv")
