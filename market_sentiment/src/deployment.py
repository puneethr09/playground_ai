from flask import Flask, request, jsonify
from preprocessing import preprocess_data
from modeling import train_model

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X, _ = preprocess_data(data["text"])
    model = train_model(X, data["labels"])
    prediction = model.predict(X)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
