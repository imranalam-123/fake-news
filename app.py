from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model & vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# Home page (UI)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Predict from UI (HTML form)
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return render_template("index.html", prediction=prediction)

# API endpoint (JSON)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json
    text = data.get("text", "")

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
