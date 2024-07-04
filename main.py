import joblib
from flask import Flask, request, jsonify
import numpy as np

app=Flask(__name__)

model1 = joblib.load('my_model1.joblib')
model2 = joblib.load('my_model2.joblib')

@app.route('/')
def index():
    return '<h1>Hello World</h1>'

@app.route('/predict-backlinks', methods=['POST'])
def predict() :
    data = request.get_json(force=True)
    input_data = np.array(data['input'])
    prediction = model1.predict(input_data)
    output = prediction.tolist()
    return jsonify({'prediction ' : output })

@app.route('/predict-linking-domains', methods=['POST'])
def predict_link() :
    data = request.get_json(force=True)
    input_data = np.array(data['input'])
    prediction = model2.predict(input_data)
    output = prediction.tolist()
    return jsonify({'prediction ' : output })

app.run()