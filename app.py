from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("models", "iris_nn_model.h5")
model = load_model(MODEL_PATH)

# Load scaler (if used)
SCALER_PATH = os.path.join("models", "scaler.save")
scaler = joblib.load(SCALER_PATH)

# Class labels
classes = ["Setosa", "Versicolor", "Virginica"]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])

        # Convert to numpy array
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale input
        sample = scaler.transform(sample)

        # Prediction
        pred = model.predict(sample)
        class_id = np.argmax(pred)
        prediction = classes[class_id]

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    
    if __name__ == "__main__":
       import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)