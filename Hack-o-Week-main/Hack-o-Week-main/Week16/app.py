from flask import Flask, request, jsonify
import numpy as np
import pickle
from cryptography.fernet import Fernet

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Encryption setup
key = Fernet.generate_key()
cipher = Fernet(key)

HIGH_BPM_THRESHOLD = 110

@app.route("/")
def home():
    return "Server Running"

@app.route("/stream", methods=["POST"])
def stream():
    data = request.json
    bpm = data.get("heart_rate")

    prediction = model.predict(np.array([[bpm]]))[0]
    is_anomaly = (prediction == -1) or (bpm > HIGH_BPM_THRESHOLD)

    response = {
        "heart_rate": bpm,
        "status": "Anomaly" if is_anomaly else "Normal"
    }

    if is_anomaly:
        msg = f"ALERT: {bpm} BPM detected"
        encrypted = cipher.encrypt(msg.encode()).decode()
        response["encrypted_alert"] = encrypted

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)