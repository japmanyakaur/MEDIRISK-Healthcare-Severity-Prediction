"""
=============================================================
  This API loads the trained Random Forest model and serves
  predictions via REST endpoints that Mendix calls.

  HOW TO RUN LOCALLY:
    pip install flask flask-cors scikit-learn pandas numpy
    python app.py
    → API runs at http://localhost:5000

  ENDPOINTS:
    GET  /health         → check if API is running
    POST /predict        → get crowd level prediction
    GET  /stats          → get dashboard statistics
=============================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Load model on startup ────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'smart_triage_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    artifacts = pickle.load(f)

model      = artifacts['model']
le_gender  = artifacts['le_gender']
le_race    = artifacts['le_race']
le_dept    = artifacts['le_dept']
le_target  = artifacts['le_target']
classes    = artifacts['classes']

print(f" Model loaded — classes: {classes}")

# ── Helper: safe label encode ────────────────────────────────
def safe_encode(encoder, value, default=0):
    try:
        return int(encoder.transform([str(value)])[0])
    except:
        return default

# ── Routes ───────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'Random Forest — Smart Triage ED Optimizer',
        'accuracy': '93.70%',
        'classes': classes
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts patient + hospital context data, returns crowd level prediction.

    Expected JSON body:
    {
        "Age": 35,
        "Gender": "M",
        "Race": "White",
        "Department_Referral": "General Practice",
        "Hour_of_Day": 19,
        "Day_of_Week": 4,
        "Month": 3,
        "Is_Weekend": 0,
        "Is_Peak_Hour": 1,
        "Staff_Count": 8,
        "Patients_In_Queue": 18,
        "Available_Beds": 5,
        "Pending_Labs": 9,
        "Triage_Level": 2,
        "Satisfaction_Score": 5.0,
        "Admitted": 0
    }

    Returns:
    {
        "crowd_level": "High",
        "confidence": 87.5,
        "wait_time_estimate": "45-60 mins",
        "recommendation": "Call in additional staff immediately",
        "alert_color": "red",
        "probabilities": {"High": 0.875, "Low": 0.05, "Medium": 0.075}
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        # Build feature vector
        features = [
            float(data.get('Age', 35)),
            safe_encode(le_gender, data.get('Gender', 'M')),
            safe_encode(le_race,   data.get('Race', 'White')),
            safe_encode(le_dept,   data.get('Department_Referral', 'None')),
            float(data.get('Hour_of_Day', 12)),
            float(data.get('Day_of_Week', 2)),
            float(data.get('Month', 6)),
            float(data.get('Is_Weekend', 0)),
            float(data.get('Is_Peak_Hour', 0)),
            float(data.get('Staff_Count', 12)),
            float(data.get('Patients_In_Queue', 8)),
            float(data.get('Available_Beds', 20)),
            float(data.get('Pending_Labs', 5)),
            float(data.get('Triage_Level', 3)),
            float(data.get('Satisfaction_Score', 5.0)),
            float(data.get('Admitted', 0)),
        ]

        X = np.array(features).reshape(1, -1)
        pred       = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        crowd_level = le_target.inverse_transform([pred])[0]
        confidence  = round(float(pred_proba.max()) * 100, 1)

        # Build probability dict
        probs = {
            cls: round(float(p), 3)
            for cls, p in zip(classes, pred_proba)
        }

        # Business logic based on prediction
        if crowd_level == 'High':
            wait_estimate  = '45-60+ mins'
            recommendation = 'CRITICAL: Call in additional staff immediately. Consider diverting non-urgent cases.'
            alert_color    = 'red'
        elif crowd_level == 'Medium':
            wait_estimate  = '25-45 mins'
            recommendation = 'MODERATE: Monitor closely. Consider activating surge protocol.'
            alert_color    = 'amber'
        else:
            wait_estimate  = '10-25 mins'
            recommendation = 'NORMAL: Current staffing is adequate. Continue monitoring.'
            alert_color    = 'green'

        return jsonify({
            'crowd_level':      crowd_level,
            'confidence':       confidence,
            'wait_time_estimate': wait_estimate,
            'recommendation':   recommendation,
            'alert_color':      alert_color,
            'probabilities':    probs,
            'is_peak_hour':     bool(data.get('Is_Peak_Hour', 0)),
            'staff_count':      int(data.get('Staff_Count', 12)),
            'patients_in_queue': int(data.get('Patients_In_Queue', 8))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Returns current dashboard statistics for Mendix to display."""
    from datetime import datetime
    import random
    random.seed(int(datetime.now().strftime('%H')))

    hour = datetime.now().hour
    is_peak = 1 if (6 <= hour < 10) or (18 <= hour < 24) else 0
    is_weekend = 1 if datetime.now().weekday() >= 5 else 0

    if 8 <= hour < 16:
        staff = random.randint(12, 18)
    elif 16 <= hour < 24:
        staff = random.randint(8, 13)
    else:
        staff = random.randint(4, 8)

    queue = random.randint(2, 5)
    if is_peak:
        queue += random.randint(10, 20)
    if is_weekend:
        queue += random.randint(4, 8)

    beds = random.randint(15, 40)
    if is_peak:
        beds = max(2, beds - random.randint(10, 18))

    labs = random.randint(2, 6)
    if is_peak:
        labs += random.randint(3, 8)

    features = [35, 1, 5, 0, hour, datetime.now().weekday(),
                datetime.now().month, is_weekend, is_peak,
                staff, queue, beds, labs, 3, 5.0, 0]

    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    crowd_level = le_target.inverse_transform([pred])[0]
    confidence = round(float(pred_proba.max()) * 100, 1)

    if crowd_level == 'High':
        wait_time = random.randint(45, 65)
        alert_color = 'red'
    elif crowd_level == 'Medium':
        wait_time = random.randint(25, 44)
        alert_color = 'amber'
    else:
        wait_time = random.randint(10, 24)
        alert_color = 'green'

    return jsonify({
        'total_patients': 9216,
        'crowd_level': crowd_level,
        'confidence': confidence,
        'wait_time_mins': wait_time,
        'staff_count': staff,
        'patients_in_queue': queue,
        'available_beds': beds,
        'pending_labs': labs,
        'is_peak_hour': bool(is_peak),
        'alert_color': alert_color,
        'current_hour': hour,
        'model_accuracy': 93.70
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'app': 'Smart-Triage ED Optimizer API',
        'version': '1.0',
        'endpoints': ['/health', '/predict', '/stats']
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)