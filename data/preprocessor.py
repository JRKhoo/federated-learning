import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import argparse

class DataPreprocessor:

    ID_COLUMNS = [
        'encounter_id', 'patient_nbr'
    ]

    COLUMNS_TO_DROP = [
        'encounter_id', 'patient_nbr', 'weight', 'max_glu_serum', 'A1Cresult'
    ]

    CATEGORICAL_COLUMNS = [
        'race', 'gender', 'age',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'payer_code', 'medical_specialty',
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
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    # clean, encode and save data
    # input path: path to raw data csv
    # output path: path to save cleaned data csv
    # encoder_path: path to save/load encoders
    # fit_encoders: if True, create new encoders. if False, load existing
    def preprocess_data(self, input_path, output_path, encoder_path=None, 
                       fit_encoders=True):
        
        print(f"Preprocessing: {os.path.basename(input_path)}")
        
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # preserve IDs for reference (will add back at the end)
        ids_df = df[self.ID_COLUMNS].copy() if all(col in df.columns for col in self.ID_COLUMNS) else None
        
        # drop ID columns AND weight, max_glu_serum, A1Cresult
        print(f"\nDropping columns: {', '.join(self.COLUMNS_TO_DROP)}")
        df = df.drop(columns=[col for col in self.COLUMNS_TO_DROP if col in df.columns])
        
        # separate target variable
        target = None
        if 'readmitted' in df.columns:
            target = df['readmitted'].copy()
            df = df.drop('readmitted', axis=1)
            # print(f"Separated target variable 'readmitted'")
        
        # replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # remove rows with missing values
        initial_count = len(df)
        mask = ~df.isna().any(axis=1)
        df = df[mask]
        if ids_df is not None:
            ids_df = ids_df[mask]
        if target is not None:
            target = target[mask]
        
        removed_count = initial_count - len(df)
        print(f"Removed {removed_count} rows with missing values")
        print(f"Remaining: {len(df)} rows")
        
        # identify categorical and numerical columns
        categorical_cols = [col for col in self.CATEGORICAL_COLUMNS if col in df.columns]
        integer_cols = [col for col in self.INTEGER_COLUMNS if col in df.columns]
        
        print(f"Categorical columns: {len(categorical_cols)}")
        print(f"Integer columns: {len(integer_cols)}")
        
        # load/create encoders
        if not fit_encoders and encoder_path and os.path.exists(encoder_path):
            print(f"Using existing encoders from {encoder_path}")
            encoder_data = joblib.load(encoder_path)
            self.label_encoders = encoder_data['label_encoders']
            self.scaler = encoder_data['scaler']
        else:
            print(f"\nCreating new encoders")
            
        # encode categorical columns
        print("\nEncoding categorical variables...")
        for col in categorical_cols:
            
            # create new encoder
            if fit_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            
            # use existing encoder
            else:
                if col in self.label_encoders:
                    # handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: x if str(x) in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    print(f"No encoder found for {col}, creating new one")
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        print(f"Encoded {len(categorical_cols)} categorical columns")
        
        # scale integer columns
        if len(integer_cols) > 0:
            print(f"\nScaling integer variables...")
            if fit_encoders:
                df[integer_cols] = self.scaler.fit_transform(df[integer_cols])
            else:
                df[integer_cols] = self.scaler.transform(df[integer_cols])
            print(f"Scaled {len(integer_cols)} integer columns")
        
        # save encoders
        if fit_encoders and encoder_path:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            joblib.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'categorical_columns': categorical_cols,
                'integer_columns': integer_cols
            }, encoder_path)
            print(f"\nEncoder saved to {encoder_path}")
        
        # add back IDs and target if requested
        result_df = df.copy()
        if ids_df is not None:
            result_df = pd.concat([ids_df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
        
        if target is not None:
            # convert target to binary
            target_binary = target.apply(lambda x: 0 if x == 'NO' else 1)
            result_df['readmitted'] = target_binary.values
        
        # save preprocessed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}\n")
        
        return result_df


def main():
    parser = argparse.ArgumentParser(description='Preprocess hospital data for federated learning')
    parser.add_argument('--input_file', type=str, default='raw/diabetic_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_file', type=str, default='cleaned/cleaned_data.csv',
                        help='Path to saved preprocessed CSV file')
    parser.add_argument('--encoder_path', type=str, default='encoders/encoders.pkl',
                        help='Path to save/load encoders')
    parser.add_argument('--fit_encoders', action='store_false',
                        help='Create new encoders (default: use existing encoders)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found")
        return

    preprocessor = DataPreprocessor()
    
    print(f"Running preprocessor with following arguments...")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Encoders: {args.encoder_path}")
    print(f"Fit encoders: {args.fit_encoders}\n")
    
    # preprocess the data file and create encoders
    preprocessor.preprocess_data(
        input_path=args.input_file,
        output_path=args.output_file,
        encoder_path=args.encoder_path,
        fit_encoders=args.fit_encoders
    )
    
    print(f"Preprocessing Complete")
    print(f"Preprocessed file saved: {args.output_file}")
    print(f"Encoders saved: {args.encoder_path}")


if __name__ == '__main__':
    main()
