import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    # Convert prediction to a standard Python type
    prediction = int(prediction)  # or float(prediction) if your prediction is a float
    
    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5008)

