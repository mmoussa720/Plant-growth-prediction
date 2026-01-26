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

WATER_ORDER = ["weekly", "bi-weekly", "daily"]

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
            'date': firestore.SERVER_TIMESTAMP
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
