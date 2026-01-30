import csv
from flask import Flask, request, jsonify
import pickle
from dotenv import load_dotenv
import pandas as pd
import os
from firebase_admin import credentials, initialize_app, firestore

load_dotenv()

private_key = os.getenv("private_key")

firebase_config = {
    "type": "service_account",
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    "private_key": private_key.replace("\\n", "\n"),
    "client_email": os.getenv("client_email"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
    "token_uri": "https://oauth2.googleapis.com/token"
}

cred = credentials.Certificate(firebase_config)
initialize_app(cred)
db = firestore.client()

model = pickle.load(open('plant_growth_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

app = Flask(__name__)

CATEGORICAL_COLS = ["Soil_Type", "Water_Frequency", "Fertilizer_Type"]

WATER_ORDER = ["bi-weekly","weekly", "daily"]

MAX_STEPS = 6 

def encode_input(df):
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])
    return df


def improve_conditions(data):
    improvements = []
    current_data = data.copy()

    for _ in range(MAX_STEPS):

        test_df = encode_input(pd.DataFrame([current_data]))
        prediction = model.predict(test_df)[0]

        if prediction == 1:
            return 1, list(set(improvements)), current_data

        current_water = current_data["Water_Frequency"]
        idx = WATER_ORDER.index(current_water)
        if idx < len(WATER_ORDER) - 1:
            current_data["Water_Frequency"] = WATER_ORDER[idx + 1]
            improvements.append("Increase watering frequency")

        if current_data["Temperature"] > 26:
            current_data["Temperature"] -= 2
            improvements.append("Reduce temperature")

        if current_data["Humidity"] < 60:
            current_data["Humidity"] += 5
            improvements.append("Increase humidity")

        if current_data["Sunlight_Hours"] > 8:
            current_data["Sunlight_Hours"] -= 1
            improvements.append("Reduce sunlight exposure")

    return 0, list(set(improvements)), current_data

@app.route('/settings', methods=['GET'])
def get_settings():
    try:
        configs = db.collection('configurations').limit(1).stream()
        for doc in configs:
            return jsonify(doc.to_dict())
        return jsonify({})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prediction', methods=['GET'])
def get_prediction():
    try:
        predictions = db.collection('predictions').order_by('date').limit(10).stream()
        return jsonify([doc.to_dict() for doc in predictions])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_df = encode_input(pd.DataFrame([data]))
        prediction = model.predict(input_df)[0]

        result = {
            "growth_prediction": int(prediction),
            "improvements_needed": [],
            "recommended_final_conditions": {}
        }

        if prediction == 0:
            new_pred, improvements, final_conditions = improve_conditions(data)
            result["improvements_needed"] = improvements
            result["recommended_final_conditions"] = final_conditions

        db.collection('predictions').add({
            'input_data': data,
            'predicted_growth_milestone': int(prediction),
            'improvements_needed': result["improvements_needed"],
            'recommended_final_conditions': result["recommended_final_conditions"],
            'date': firestore.SERVER_TIMESTAMP
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/configure', methods=['POST'])
def configure():
    try:
        data = request.json
        db.collection('configurations').add({
            'soil_type': data['soil_type'],
            'fertilizer_Type': data['fertilizer_type'],
            'water_Frequency': data['water_frequency'],
            'watering_day':data['watering_day'],
            'watering_time':data['watering_time'],
        })

        return jsonify({"status": "Configuration saved successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def load_soil_data(csv_file="irrigation_data.csv"):
    soil_dict = {}
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            soil_type = row["Soil_Type"]
            soil_dict[soil_type] = {
                "FC": float(row["Field_Capacity"]),
                "WP": float(row["Wilting_Point"]),
                "50%": float(row["Depletion_50"])
            }
    return soil_dict

soil_data = load_soil_data()

@app.route("/check_irrigation", methods=["POST"])
def check_irrigation():
    data = request.get_json()
    soil_type = data.get("soil_type")
    vwc = data.get("vwc") 

    if soil_type not in soil_data:
        return jsonify({"error": "Unknown soil type"}), 400

    soil = soil_data[soil_type]
    # Check if VWC is below 50% depletion threshold
    if vwc < soil["50%"]:
        return jsonify({"need_water": 1})
    else:
        return jsonify({"need_water": 0})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


