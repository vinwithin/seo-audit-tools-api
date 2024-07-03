from flask import Flask, request, jsonify
import joblib
import numpy as np


app=Flask(__name__)

model1 = joblib.load('my_model1.joblib')
model2 = joblib.load('my_model2.joblib')

@app.route('/predict-backlinks', methods=['POST'])
def predict() :
    data = request.get_json(force=True)
    input_data = np.array(data['input'])
    prediction = model1.predict(input_data)
    output = prediction.tolist()
    return jsonify({'prediction ' : output })

if __name__ == '__main__':
    app.run(debug=true)