from flask import Flask, request, jsonify
import pickle
import pandas as pd

with open('plant_growth_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    for col in ['Soil_Type', 'Water_Frequency', 'Fertilizer_Type']:
        df[col] = label_encoders[col].transform(df[col])

    prediction = int(model.predict(df)[0])
    return jsonify({'Growth_Milestone': prediction})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
