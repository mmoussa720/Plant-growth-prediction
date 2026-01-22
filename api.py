from flask import Flask, request, jsonify
import pickle
from dotenv import load_dotenv
import pandas as pd
import os
from firebase_admin import credentials, initialize_app,firestore
load_dotenv()

private_key = os.getenv("private_key")
if private_key is None:
    raise ValueError("private_key not found in environment variables!")
firebase_config = {
    "type": "service_account",
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    "private_key": private_key.replace("\\n", "\n"),
    "client_email": os.getenv("client_email"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
    "token_uri": "https://oauth2.googleapis.com/token"
}
cred=credentials.Certificate(firebase_config)
initialize_app(cred)
db=firestore.client()
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
    db.collection('predictions').add({
        'input_data': data,
        'date': firestore.SERVER_TIMESTAMP,
        'predicted_growth_milestone': prediction
    })
    return jsonify({'Growth_Milestone': prediction})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
