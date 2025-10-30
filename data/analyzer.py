import pandas as pd
import numpy as np
import argparse
import os

def analyze_missing_values(data_path):
    """
    Analyze missing values in the dataset and display statistics.
    
    Args:
        data_path: Path to the CSV file
    """
    print(f"\n{'='*80}")
    print(f"Missing Values Analysis: {os.path.basename(data_path)}")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Replace '?' with NaN for proper missing value detection
    df_clean = df.replace('?', np.nan)
    
    # Calculate missing values per column
    missing_stats = pd.DataFrame({
        'Column': df_clean.columns,
        'Missing_Count': df_clean.isnull().sum(),
        'Missing_Percentage': (df_clean.isnull().sum() / len(df_clean)) * 100,
        'Data_Type': df_clean.dtypes
    })
    
    # Sort by missing percentage (descending)
    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
    
    # Overall statistics
    total_cells = df_clean.shape[0] * df_clean.shape[1]
    total_missing = df_clean.isnull().sum().sum()
    overall_missing_pct = (total_missing / total_cells) * 100
    
    print(f"OVERALL STATISTICS:")
    print(f"{'-'*80}")
    print(f"Total cells: {total_cells:,}")
    print(f"Total missing values: {total_missing:,}")
    print(f"Overall missing percentage: {overall_missing_pct:.2f}%\n")
    
    # Rows with at least one missing value
    rows_with_missing = df_clean.isnull().any(axis=1).sum()
    rows_with_missing_pct = (rows_with_missing / len(df_clean)) * 100
    
    print(f"ROWS WITH MISSING VALUES:")
    print(f"{'-'*80}")
    print(f"Rows with at least one missing value: {rows_with_missing:,} ({rows_with_missing_pct:.2f}%)")
    print(f"Complete rows (no missing values): {len(df_clean) - rows_with_missing:,} ({100 - rows_with_missing_pct:.2f}%)\n")
    
    # Column statistics
    print(f"MISSING VALUES BY COLUMN:")
    print(f"{'-'*80}")
    
    # Separate columns with and without missing values
    cols_with_missing = missing_stats[missing_stats['Missing_Count'] > 0]
    cols_without_missing = missing_stats[missing_stats['Missing_Count'] == 0]
    
    if len(cols_with_missing) > 0:
        print(f"\nColumns WITH missing values ({len(cols_with_missing)} columns):")
        print(f"{'-'*80}")
        print(f"{'Column':<30} {'Missing':<12} {'Percentage':<12} {'Type':<15}")
        print(f"{'-'*80}")
        
        for _, row in cols_with_missing.iterrows():
            print(f"{row['Column']:<30} {int(row['Missing_Count']):<12} {row['Missing_Percentage']:>10.2f}% {str(row['Data_Type']):<15}")
    
    if len(cols_without_missing) > 0:
        print(f"\n\nColumns WITHOUT missing values ({len(cols_without_missing)} columns):")
        print(f"{'-'*80}")
        print(", ".join(cols_without_missing['Column'].tolist()))
    
    # Category analysis
    print(f"\n\nMISSING VALUES BY CATEGORY:")
    print(f"{'-'*80}")
    
    # Define column categories from your schema
    categorical_cols = [
        'race', 'gender', 'age', 'weight',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3',
        'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
        'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed'
    ]
    
    integer_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses'
    ]
    
    # Calculate missing by category
    for category_name, columns in [('Categorical', categorical_cols), ('Integer', integer_cols)]:
        cols_in_data = [col for col in columns if col in df_clean.columns]
        if cols_in_data:
            category_missing = df_clean[cols_in_data].isnull().sum().sum()
            category_total = len(df_clean) * len(cols_in_data)
            category_pct = (category_missing / category_total) * 100
            
            print(f"\n{category_name} columns ({len(cols_in_data)} columns):")
            print(f"  Total missing: {category_missing:,} / {category_total:,} ({category_pct:.2f}%)")
    
    print(f"\n{'='*80}\n")
    
    return missing_stats


def main():
    parser = argparse.ArgumentParser(description='Analyze missing values in dataset')
    parser.add_argument('--input_file', type=str, default='raw/diabetic_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--save_report', action='store_true',
                        help='Save detailed report to CSV')
    parser.add_argument('--output_path', type=str, default='data/missing_values_report.csv',
                        help='Path to save report CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found - {args.input_file}")
        return
    
    # Analyze missing values
    missing_stats = analyze_missing_values(args.input_file)
    
    # Save report if requested
    if args.save_report:
        missing_stats.to_csv(args.output_path, index=False)
        print(f"Detailed report saved to: {args.output_path}")


if __name__ == '__main__':
    main()