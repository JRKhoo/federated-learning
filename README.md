# Federate Learning Simulation

Simulation of federated learning to train a model to predict readmission rates of diabetic patients.

## Setup
Initialize virtual environment (first time setup)
```bash
python -m venv .venv
```

Activate virtual environment (subsequent use)
```bash
.venv\Scripts\Activate
```

Install requirements
```bash
pip install -r requirements.txt
```

## Data
This project uses the "Diabetes 130‑US hospitals for years 1999-2008" dataset from the UCI Machine Learning Repository:
https://archive-beta.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains roughly 100k hospital admission records (patient encounters) for diabetic patients collected across 130 US hospitals. Each row is an encounter with demographic, admission/discharge, diagnosis, lab test, procedure and medication fields (examples: race, gender, age, num_lab_procedures, number_inpatient, A1Cresult, medication fields). 

The primary target used in this project is the "readmitted" field, which indicates whether the patient was readmitted within 30 days ("<30"), after 30 days (">30"), or not readmitted ("NO").