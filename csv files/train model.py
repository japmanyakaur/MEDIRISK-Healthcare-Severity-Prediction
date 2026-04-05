
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import pickle
import os

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# STEP 1 | LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  Smart-Triage Model Training")
print("=" * 55)

try:
    df = pd.read_csv(r"C:\Users\Admin\Desktop\xpecto\csv files\hospital_er_rapidminer.csv")
    print(f"\n Loaded — {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    raise FileNotFoundError(
        "\n hospital_er_rapidminer.csv not found.\n"
        "   Place this script in the same folder as the CSV."
    )

print(f"   Crowd_Level distribution:")
print(df['Crowd_Level'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2 | ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────
print("\nEncoding categorical columns...")

le_gender = LabelEncoder()
le_race   = LabelEncoder()
le_dept   = LabelEncoder()

df['Gender_enc'] = le_gender.fit_transform(df['Gender'].fillna('Unknown'))
df['Race_enc']   = le_race.fit_transform(df['Race'].fillna('Unknown'))
df['Dept_enc']   = le_dept.fit_transform(
    df['Department_Referral'].fillna('None')
)

print(f"    Gender classes : {list(le_gender.classes_)}")
print(f"    Race classes   : {list(le_race.classes_)}")
print(f"    Dept classes   : {list(le_dept.classes_)}")

# ─────────────────────────────────────────────────────────────
# STEP 3 | DEFINE FEATURES
# ─────────────────────────────────────────────────────────────
feature_cols = [
    'Age', 'Gender_enc', 'Race_enc', 'Dept_enc',
    'Hour_of_Day', 'Day_of_Week', 'Month',
    'Is_Weekend', 'Is_Peak_Hour',
    'Staff_Count', 'Patients_In_Queue',
    'Available_Beds', 'Pending_Labs',
    'Triage_Level', 'Satisfaction_Score', 'Admitted'
]

X = df[feature_cols]
y = df['Crowd_Level']

le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

print(f"\n   Features ({len(feature_cols)}): {feature_cols}")
print(f"   Target classes: {list(le_target.classes_)}")

# ─────────────────────────────────────────────────────────────
# STEP 4 | TRAIN RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("\n Training Random Forest (500 trees)...")
print("   This may take 1-2 minutes...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# Cross validation — same as RapidMiner's 10-fold
scores = cross_val_score(model, X, y_enc, cv=10, scoring='accuracy')
print(f"\n    10-fold CV accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# Train final model on full data
model.fit(X, y_enc)
print(f"    Model trained on {len(X)} samples")

# ─────────────────────────────────────────────────────────────
# STEP 5 | TEST PREDICTION
# ─────────────────────────────────────────────────────────────
print("\n Testing prediction...")

test_input = {
    'Age': 35,
    'Gender': 'M',
    'Race': 'White',
    'Department_Referral': 'General Practice',
    'Hour_of_Day': 19,
    'Day_of_Week': 4,
    'Month': 3,
    'Is_Weekend': 0,
    'Is_Peak_Hour': 1,
    'Staff_Count': 8,
    'Patients_In_Queue': 18,
    'Available_Beds': 5,
    'Pending_Labs': 9,
    'Triage_Level': 2,
    'Satisfaction_Score': 5.0,
    'Admitted': 1
}

row = [
    test_input['Age'],
    int(le_gender.transform([test_input['Gender']])[0]),
    int(le_race.transform([test_input['Race']])[0]),
    int(le_dept.transform([test_input['Department_Referral']])[0]),
    test_input['Hour_of_Day'], test_input['Day_of_Week'],
    test_input['Month'], test_input['Is_Weekend'],
    test_input['Is_Peak_Hour'], test_input['Staff_Count'],
    test_input['Patients_In_Queue'], test_input['Available_Beds'],
    test_input['Pending_Labs'], test_input['Triage_Level'],
    test_input['Satisfaction_Score'], test_input['Admitted']
]

pred        = model.predict([row])
pred_proba  = model.predict_proba([row])
crowd_level = le_target.inverse_transform(pred)[0]
confidence  = round(float(pred_proba[0].max()) * 100, 1)

print(f"   Input: Peak hour, 18 patients in queue, 8 staff, 5 beds")
print(f"   Prediction: {crowd_level} (confidence: {confidence}%)")

# ─────────────────────────────────────────────────────────────
# STEP 6 | SAVE MODEL
# ─────────────────────────────────────────────────────────────
print("\n Saving model...")

artifacts = {
    'model':        model,
    'le_gender':    le_gender,
    'le_race':      le_race,
    'le_dept':      le_dept,
    'le_target':    le_target,
    'feature_cols': feature_cols,
    'classes':      list(le_target.classes_)
}

output_path = r"C:\Users\Admin\Desktop\xpecto\smart-triage-api\smart_triage_model.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(artifacts, f)

size_mb = os.path.getsize(output_path) / (1024 * 1024)

print(f"""
{'='*55}
  DONE — saved: {output_path}
  Size    : {size_mb:.1f} MB
  Accuracy: {scores.mean()*100:.2f}%
  Classes : {list(le_target.classes_)}
{'='*55}

  NEXT STEP:
  Put smart_triage_model.pkl in your smart-triage-api folder
  alongside app.py and run:  python app.py
{'='*55}
""")