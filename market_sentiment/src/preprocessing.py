import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["title"])
    return X, df["title"]


if __name__ == "__main__":
    X, y = preprocess_data("../data/raw/Apple_articles.csv")
    # Save processed data if needed
