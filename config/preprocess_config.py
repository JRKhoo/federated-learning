COLUMNS_TO_DROP = [
    'encounter_id', 'patient_nbr', 'weight', 'max_glu_serum', 'A1Cresult', 'payer_code'
]

CATEGORICAL_COLUMNS = [
    'race', 'gender', 'age',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'medical_specialty',
    'diag_1', 'diag_2', 'diag_3',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone', 'change', 'diabetesMed'
]

INTEGER_COLUMNS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]