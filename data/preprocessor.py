import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import argparse
from pathlib import Path

# add root so config is discoverable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import config.preprocess_config as pp_config

class DataPreprocessor:

    COLUMNS_TO_DROP = pp_config.COLUMNS_TO_DROP
    CATEGORICAL_COLUMNS = pp_config.CATEGORICAL_COLUMNS
    INTEGER_COLUMNS = pp_config.INTEGER_COLUMNS
    
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
        
        # drop ID columns AND weight, max_glu_serum, A1Cresult
        print(f"\nDropping columns: {', '.join(self.COLUMNS_TO_DROP)}")
        df = df.drop(columns=[col for col in self.COLUMNS_TO_DROP if col in df.columns])
        
        # separate target variable
        target = None
        if 'readmitted' in df.columns:
            target = df['readmitted'].copy()
            df = df.drop('readmitted', axis=1)
        
        # replace '?' with NaN
        #df = df.replace('?', np.nan)
        
        # remove rows with missing values
        # initial_count = len(df)
        # mask = ~df.isna().any(axis=1)
        # df = df[mask]
        # if target is not None:
        #     target = target[mask]

        df = df.replace('?', np.nan)

        # identify categorical and numerical columns
        categorical_cols = [col for col in self.CATEGORICAL_COLUMNS if col in df.columns]

        # everything else that is number → scale it
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # fill numeric with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # fill categorical with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        #removed_count = initial_count - len(df)
        #print(f"Removed {removed_count} rows with missing values")
        #print(f"Remaining: {len(df)} rows")

        # but remove target from numeric cols if present
        if 'readmitted' in numeric_cols:
            numeric_cols.remove('readmitted')
        
        print(f"Categorical columns: {len(categorical_cols)}")
        print(f"Numeric columns: {len(numeric_cols)}")
        
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
        
        if len(numeric_cols) > 0:
            print(f"\nScaling numeric variables...")
            if fit_encoders:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            print(f"Scaled {len(numeric_cols)} numeric columns")
        
        # save encoders
        if fit_encoders and encoder_path:
            os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
            joblib.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'categorical_columns': categorical_cols,
                'numeric_columns': numeric_cols
            }, encoder_path)
            print(f"\nEncoder saved to {encoder_path}")
        
        # add back target variable
        result_df = df.copy()
        
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
