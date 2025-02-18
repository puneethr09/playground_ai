import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(data_path="data/processed/processed_data.csv"):
    # Load the processed data
    processed_data = pd.read_csv(data_path)

    # Split the data into features and target
    X = processed_data.drop("target", axis=1)
    y = processed_data["target"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the model
    model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model with a specific protocol
    joblib.dump(model, "models/model.pkl", protocol=4)

    return model


if __name__ == "__main__":
    train_model()
