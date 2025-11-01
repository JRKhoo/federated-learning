# Federate Learning Simulation

Simulation of federated learning to train a model to predict readmission rates of diabetic patients.

## Directory Structure
```
federated-learning/
├─ data/
│  ├─ analyzer.py          # Analyze unprocessed data
│  ├─ preprocessor.py      # Clean, encode, scale
│  ├─ splitter.py          # Split into hospitals and test
│  ├─ raw/                 # Original data
│  ├─ cleaned/             # Preprocessed CSVs
│  ├─ split/               # Per-hospital and test CSVs
│  └─ encoders/            # Saved encoders
├─ global_model/           # Model weights (.npz)
├─ src/
│  ├─ trainer.py           # Train per hospital
│  └─ model_tester.py      # Evaluate model
├─ config/
│  ├─ mlp_config.py        # MLP params
│  └─ dp_config.py         # DP noise params
├─ requirements.txt
└─ README.md
```
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

## Training
We are using the Scikit-learn framework to implement a multilayer perceptron model.

### Execution
Navigate to the root directory (if you are not already there)<br>

Execute the trainer, passing in path to data file to train on: <br>
`python src/trainer.py <path to data file>`

Example execution:
```bash
python src/trainer.py data/split/hospital1.csv
```

Model weights will be generated in `.npz` numpy format and stored in `global_model` directory. <br>
Evaluation of the model is done automatically against `test_data.csv`.

### Model Tuning
Tuning of the multilayer perceptron is done in the `config/mlp_config.py` file. <br>
Tuning of differential privacy noise is done in the `config/dp_config.py` file. <br>
Changes will be reflected upon execution of `trainer.py`.

## Evaluation
We evaluate the model using 4 main metrics:
- `Accuracy`: Proportion of correct predictions (both positive and negative)
- `Precision`: Of all predicted readmissions, what proportion was actually readmitted?
- `Recall`: Of all actual readmissions, what proportion did the model correctly identify?
- `F1 Score`: Balance of precision and recall

### Execution
You can run the evaluation independently of training against `test_data.csv`.<br>
**!!! Use the same model parameters for evaluation as used in training !!!** <br>


Navigate to the root directory (if you are not already there)<br>

Execute the evaluator, passing in path to generated model weights: <br>
`python src/model_tester.py <path to model weights>`
Evaluator expects weights in `.npz` numpy format.

Example execution:
```bash
python src/model_tester.py global_model/hospital1.npz
```

## Data
This project uses the "Diabetes 130‑US hospitals for years 1999-2008" dataset from the UCI Machine Learning Repository:
https://archive-beta.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains roughly 100k hospital admission records (patient encounters) for diabetic patients collected across 130 US hospitals. Each row is an encounter with demographic, admission/discharge, diagnosis, lab test, procedure and medication fields (examples: race, gender, age, num_lab_procedures, number_inpatient, A1Cresult, medication fields). 

The primary target used in this project is the "readmitted" field, which indicates whether the patient was readmitted within 30 days ("<30"), after 30 days (">30"), or not readmitted ("NO").

### Preprocessed Data
Data has already been preprocessed according to the following logic. <br>

The following fields have been dropped:
- `encounter_id`: Unique identifier, no training value
- `patient_nbr`: Unique identifier, no training value
- `weight`: >50% Missing values
- `A1Cresult`: >50% Missing values
- `max_glu_serum`: >50% Missing values

For entries with missing values, we chose to drop them instead of conducting imputation, as features with missing values tended to have a high percentage of such missing values, imputation would skew the results significantly.

### Preprocessing steps
Navigate to the data directory, all data processing is done here.
```bash
cd data
```

Run the preprocessor on the data.
```bash
python preprocessor.py
```
- `--input_file`: specify input file, defaults to `raw/diabetic_data.csv`
- `--output_file`: specify output file, defaults to `cleaned/cleaned_data.csv`
- `--encoder_path`: specify encoder to use, defaults to `encoders/encoders.pkl`
- `--fit_encoders`: if specified, use existing encoder specified in `--encoder_path`, else create new encoder stored at `data/encoders/encoders.pkl`
<br> <br>

Split the cleaned data into test and training data.
```bash
python splitter.py
```

### Identification Variables
| Variable Name | Type |
|--------------|------|
| encounter_id | Integer |
| patient_nbr | Integer |

### Categorical Variables (Label Encoded)

| Variable Name | Type | Description | Example Values |
|--------------|------|-------------|----------------|
| race | Categorical | Patient's race | Caucasian, AfricanAmerican, Asian, Hispanic |
| gender | Categorical | Patient's gender | Male, Female |
| age | Categorical | Patient's age bracket | [0-10), [10-20), [20-30), ... |
| weight | Categorical | Patient's weight bracket | [0-25), [25-50), [50-75), ... |
| admission_type_id | Categorical | Type of admission | 1, 2, 3, 4, 5, 6, 7, 8 |
| discharge_disposition_id | Categorical | Where patient was discharged to | 1-28 |
| admission_source_id | Categorical | Where patient was admitted from | 1-26 |
| payer_code | Categorical | Payment method | MC, MD, HM, UN, BC, SP, etc. |
| medical_specialty | Categorical | Doctor's specialty | Cardiology, InternalMedicine, Surgery, etc. |
| diag_1 | Categorical | Primary diagnosis code | ICD-9 codes |
| diag_2 | Categorical | Secondary diagnosis code | ICD-9 codes |
| diag_3 | Categorical | Tertiary diagnosis code | ICD-9 codes |
| max_glu_serum | Categorical | Max glucose serum test result | None, Norm, >200, >300 |
| A1Cresult | Categorical | A1C test result | None, Norm, >7, >8 |
| change | Categorical | Change in diabetic medications | Yes, No |
| diabetesMed | Categorical | Diabetic medication prescribed | Yes, No |

### Medication Variables (All Categorical)

| Variable Name | Possible Values |
|--------------|-----------------|
| metformin | No, Steady, Up, Down |
| repaglinide | No, Steady, Up, Down |
| nateglinide | No, Steady, Up, Down |
| chlorpropamide | No, Steady, Up, Down |
| glimepiride | No, Steady, Up, Down |
| acetohexamide | No, Steady, Up, Down |
| glipizide | No, Steady, Up, Down |
| glyburide | No, Steady, Up, Down |
| tolbutamide | No, Steady, Up, Down |
| pioglitazone | No, Steady, Up, Down |
| rosiglitazone | No, Steady, Up, Down |
| acarbose | No, Steady, Up, Down |
| miglitol | No, Steady, Up, Down |
| troglitazone | No, Steady, Up, Down |
| tolazamide | No, Steady, Up, Down |
| examide | No, Steady, Up, Down |
| citoglipton | No, Steady, Up, Down |
| insulin | No, Steady, Up, Down |
| glyburide-metformin | No, Steady, Up, Down |
| glipizide-metformin | No, Steady, Up, Down |
| glimepiride-pioglitazone | No, Steady, Up, Down |
| metformin-rosiglitazone | No, Steady, Up, Down |
| metformin-pioglitazone | No, Steady, Up, Down |

### Integer Variables (Scaled using StandardScaler)
Scaled to have mean=0 and std=1:

| Variable Name | Type | Description | Range |
|--------------|------|-------------|-------|
| time_in_hospital | Integer | Number of days in hospital | 1-14 |
| num_lab_procedures | Integer | Number of lab tests performed | 0-132 |
| num_procedures | Integer | Number of procedures | 0-6 |
| num_medications | Integer | Number of medications | 1-81 |
| number_outpatient | Integer | Number of outpatient visits in year before | 0-42 |
| number_emergency | Integer | Number of emergency visits in year before | 0-76 |
| number_inpatient | Integer | Number of inpatient visits in year before | 0-21 |
| number_diagnoses | Integer | Number of diagnoses entered | 1-16 |