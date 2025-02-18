import joblib
import pandas as pd


def load_model(model_path="models/model.pkl"):
    """
    Loads the trained model from the specified path.
    """
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    return model


def predict(model, data):
    """
    Makes predictions using the loaded model.
    """
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Create some dummy data for testing
    new_data = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10]})

    # Make predictions
    predictions = predict(model, new_data)

    # Print the predictions
    print(predictions)
