# MEDIRISK : Healthcare Severity Prediction

<div align="center">

![MediRisk](https://img.shields.io/badge/MediRisk-Healthcare%20AI-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Altair](https://img.shields.io/badge/Altair%20AI%20Studio-ML-orange?style=flat-square)
![Mendix](https://img.shields.io/badge/Mendix-Low%20Code-003575?style=flat-square)
![Model](https://img.shields.io/badge/Model-Random%20Forest-success?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25-success?style=flat-square)


### AI-powered healthcare intelligence system for predicting ER crowd levels and patient risk

</div>

---

#  Overview

**MediRisk** is an AI-powered healthcare intelligence platform designed to help hospitals predict:

- Emergency Room crowd levels  
- Patient risk severity  
- Wait time estimation  

The system enables hospitals to **anticipate demand, prioritize patients, and improve operational efficiency**.

---

#  Demo

##  Dashboard Preview

![Dashboard](screenshots/dashboard.png)

##  Patient Risk Prediction

![Prediction](screenshots/prediction.png)

##  Analytics View

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/af4db523-132a-47e1-aa3f-9cfb5deb2f19" />


---

#  Problem Statement

Emergency rooms face critical challenges:

- Overcrowding  
- Delayed triage  
- Manual risk assessment  
- Poor resource allocation  
- No predictive analytics  

These issues lead to **long wait times and increased patient risk**.

---

#  Solution

MediRisk provides:

- Real-time ER crowd prediction  
- Patient risk classification  
- Smart triage support  
- Predictive analytics dashboard  
- Resource optimization  

---

#  Key Features

| Feature | Description |
|---------|-------------|
| AI Crowd Prediction | Forecast ER demand |
| Risk Classification | Identify high-risk patients |
| Real-Time Analytics | Instant insights |
| Smart Dashboard | Visual decision support |
| Scalable Architecture | Production-ready system |

---

#  System Architecture

```
Hospital Data
     ↓
Data Processing
     ↓
Machine Learning Model
     ↓
Prediction API
     ↓
Mendix Dashboard
```

---

#  Tech Stack

| Layer | Technology | Purpose |
|------|------------|---------|
| Machine Learning | Random Forest | Prediction |
| Data Processing | Python, Pandas | Preprocessing |
| Model Training | RapidMiner | ML pipeline |
| Model Export | PMML | Integration |
| Frontend | Mendix | Dashboard |
| Backend | Python / Mendix | API |

---

#  Machine Learning Model

| Property | Value |
|----------|-------|
| Algorithm | Random Forest |
| Accuracy | 93% |
| Model Type | Classification |
| Output | Low / Medium / High |

---

#  Dataset

| Property | Value |
|----------|-------|
| Records | 9000+ |
| Features | 11+ |
| Target | Crowd Level |
| Format | CSV |

---

#  How It Works

1. Patient data entered  
2. Data preprocessing  
3. ML model prediction  
4. Risk classification  
5. Dashboard visualization  

---

#  Project Structure

```
XPECTO/
│
├── csv files/
│   ├── Hospital ER.csv
│   ├── hospital_er_clean.csv
│   └── hospital_er_rapidminer.csv
│
├── train model.py
│
├── MediRisk/
│
├── smart-triage-api/
│   ├── app.py
│   ├── render.yaml
│   ├── requirements.txt
│   └── smart_triage_model.pkl
│
├── data_prep.ipynb
├── ER prediction model.rmp
├── feature_dataset.ipynb
└── README.md

```

---

#  Installation

Clone repository

```
git clone https://github.com/yourusername/medirisk.git
```

Navigate to folder

```
cd medirisk
```

Install dependencies

```
pip install -r requirements.txt
```

Run project

```
python app.py
```

---

# Key Features

- Real-world healthcare problem  
- End-to-end ML pipeline  
- Low-code + AI integration  
- Recruiter-friendly project  
- Production-ready architecture  

---

#  Future Scope

- Real-time hospital integration  
- Mobile application  
- Cloud deployment  
- Multi-hospital analytics  
- Deep learning integration  

---

<div align="center">

### Built for Smarter Healthcare with AI

</div>
