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


def analyze_readmission_rates(data_path):
    """
    Analyze readmission rates in the dataset.
    
    Args:
        data_path: Path to the CSV file
    """
    print(f"\n{'='*80}")
    print(f"Readmission Analysis: {os.path.basename(data_path)}")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Check if readmitted column exists
    if 'readmitted' not in df.columns:
        print("Error: 'readmitted' column not found in dataset")
        return None
    
    # Get readmission counts and percentages
    readmission_counts = df['readmitted'].value_counts()
    readmission_percentages = df['readmitted'].value_counts(normalize=True) * 100
    
    # Total patients
    total_patients = len(df)
    
    print(f"OVERALL READMISSION STATISTICS:")
    print(f"{'-'*80}")
    print(f"Total patients: {total_patients:,}\n")
    
    # Display statistics for each category
    print(f"READMISSION BREAKDOWN:")
    print(f"{'-'*80}")
    print(f"{'Category':<25} {'Count':>15} {'Percentage':>15}")
    print(f"{'-'*80}")
    
    # Define the expected categories
    categories = {
        'NO': 'Not Readmitted',
        '<30': 'Readmitted < 30 days',
        '>30': 'Readmitted > 30 days'
    }
    
    for category, description in categories.items():
        count = readmission_counts.get(category, 0)
        percentage = readmission_percentages.get(category, 0)
        print(f"{description:<25} {count:>15,} {percentage:>14.2f}%")
    
    # Summary statistics
    print(f"\n{'-'*80}")
    print(f"SUMMARY:")
    print(f"{'-'*80}")
    
    # Count of patients who were readmitted (either <30 or >30)
    readmitted_categories = ['<30', '>30']
    readmitted_count = sum(readmission_counts.get(cat, 0) for cat in readmitted_categories)
    readmitted_percentage = (readmitted_count / total_patients) * 100
    
    not_readmitted_count = readmission_counts.get('NO', 0)
    not_readmitted_percentage = (not_readmitted_count / total_patients) * 100
    
    print(f"{'Patients Readmitted (any)':<25} {readmitted_count:>15,} {readmitted_percentage:>14.2f}%")
    print(f"{'Patients Not Readmitted':<25} {not_readmitted_count:>15,} {not_readmitted_percentage:>14.2f}%")
    
    # Additional insights
    if '<30' in readmission_counts and '>30' in readmission_counts:
        early_readmit = readmission_counts['<30']
        late_readmit = readmission_counts['>30']
        
        if readmitted_count > 0:
            early_pct_of_readmit = (early_readmit / readmitted_count) * 100
            late_pct_of_readmit = (late_readmit / readmitted_count) * 100
            
            print(f"\n{'-'*80}")
            print(f"AMONG READMITTED PATIENTS:")
            print(f"{'-'*80}")
            print(f"{'Early readmission (<30 days)':<25} {early_readmit:>15,} {early_pct_of_readmit:>14.2f}%")
            print(f"{'Late readmission (>30 days)':<25} {late_readmit:>15,} {late_pct_of_readmit:>14.2f}%")
    
    print(f"\n{'='*80}\n")
    
    return {
        'total': total_patients,
        'not_readmitted': not_readmitted_count,
        'readmitted_less_30': readmission_counts.get('<30', 0),
        'readmitted_more_30': readmission_counts.get('>30', 0),
        'total_readmitted': readmitted_count
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset')
    parser.add_argument('--input_file', type=str, default='raw/diabetic_data.csv',
                        help='Path to input CSV file')
    parser.add_argument('--analysis_type', type=str, default='all',
                        choices=['missing', 'readmission', 'all'],
                        help='Type of analysis to perform')
    parser.add_argument('--save_report', action='store_true',
                        help='Save detailed report to CSV')
    parser.add_argument('--output_path', type=str, default='data/missing_values_report.csv',
                        help='Path to save report CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found - {args.input_file}")
        return
    
    # Perform requested analysis
    if args.analysis_type in ['missing', 'all']:
        # Analyze missing values
        missing_stats = analyze_missing_values(args.input_file)
        
        # Save report if requested
        if args.save_report:
            missing_stats.to_csv(args.output_path, index=False)
            print(f"Detailed missing values report saved to: {args.output_path}")
    
    if args.analysis_type in ['readmission', 'all']:
        # Analyze readmission rates
        readmission_stats = analyze_readmission_rates(args.input_file)


if __name__ == '__main__':
    main()