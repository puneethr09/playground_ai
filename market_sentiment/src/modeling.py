from sklearn.naive_bayes import MultinomialNB


def train_model(X, y):
    model = MultinomialNB()
    model.fit(X, y)
    return model


if __name__ == "__main__":
    from preprocessing import preprocess_data

    X, y = preprocess_data("../data/raw/Apple_articles.csv")
    model = train_model(X, y)
    # Save the model if needed
