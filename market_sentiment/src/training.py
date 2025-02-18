from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import joblib

nltk.download("vader_lexicon")


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["title"])  # Drop rows with missing titles

    # Add feature engineering
    df["title_length"] = df["title"].apply(len)
    df["word_count"] = df["title"].apply(lambda x: len(x.split()))

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2)
    X_tfidf = vectorizer.fit_transform(df["title"])

    # Combine TF-IDF features with engineered features
    X = pd.concat(
        [
            pd.DataFrame(X_tfidf.toarray()),
            df[["title_length", "word_count"]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Ensure all column names are strings
    X.columns = X.columns.astype(str)

    # Generate sentiment labels
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["title"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["label"] = df["sentiment"].apply(
        lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
    )

    return X, df["label"], vectorizer


def train_model(X, y):
    # Use Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


def fine_tune_model(X, y):
    # Define the parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, refit=True, verbose=2, cv=3
    )

    # fitting the model for grid search
    grid.fit(X, y)
    return grid.best_estimator_


def train_and_evaluate(file_path):
    X, y, vectorizer = preprocess_data(file_path)

    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Fine tune the model
    model = fine_tune_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model and vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")


if __name__ == "__main__":
    train_and_evaluate("../data/raw/general_news_articles.csv")
