import pandas as pd
import os

def analyze_data():
    
    # define files
    split_dir = "split"
    raw_path = "raw/diabetic_data.csv"
    split_files = ['hospital1.csv', 'hospital2.csv', 'hospital3.csv', 'test_data.csv']
    
    print("="*80)
    print("CLEANED DATA ANALYSIS")
    print("="*80)
    print()
    
    # identify dropped features
    raw_df = pd.read_csv(raw_path, nrows=1)
    raw_features = set(raw_df.columns)
    
    split_df_sample = pd.read_csv(os.path.join(split_dir, split_files[0]), nrows=1)
    split_features = set(split_df_sample.columns)
    
    dropped_features = raw_features - split_features
    
    print("DROPPED FEATURES:")
    print("-" * 80)
    if dropped_features:
        for i, feature in enumerate(sorted(dropped_features), 1):
            print(f"{i}. {feature}")
    else:
        print("No features were dropped")
    print()
    
    # individual file analysis
    print("="*80)
    print("PER FILE STATISTICS:")
    print("="*80)
    print()
    
    total_entries = 0
    total_readmitted = 0
    
    file_stats = []
    
    for filename in split_files:
        filepath = os.path.join(split_dir, filename)
        df = pd.read_csv(filepath)
        
        num_entries = len(df)
        num_readmitted = df['readmitted'].sum()
        num_not_readmitted = num_entries - num_readmitted
        
        pct_readmitted = (num_readmitted / num_entries) * 100
        pct_not_readmitted = (num_not_readmitted / num_entries) * 100
        
        # store for totals
        total_entries += num_entries
        total_readmitted += num_readmitted
        
        file_stats.append({
            'filename': filename,
            'entries': num_entries,
            'readmitted': num_readmitted,
            'not_readmitted': num_not_readmitted,
            'pct_readmitted': pct_readmitted,
            'pct_not_readmitted': pct_not_readmitted
        })
        
        print(f"File: {filename}")
        print(f"  Number of entries: {num_entries:,}")
        print(f"  Readmitted: {num_readmitted:,} ({pct_readmitted:.2f}%)")
        print(f"  Not readmitted: {num_not_readmitted:,} ({pct_not_readmitted:.2f}%)")
        print()
    
    # calculate overall statistics
    total_not_readmitted = total_entries - total_readmitted
    total_pct_readmitted = (total_readmitted / total_entries) * 100
    total_pct_not_readmitted = (total_not_readmitted / total_entries) * 100
    
    print("="*80)
    print("OVERALL STATISTICS (ALL 4 FILES COMBINED):")
    print("="*80)
    print()
    print(f"Total number of entries: {total_entries:,}")
    print(f"Total readmitted: {total_readmitted:,} ({total_pct_readmitted:.2f}%)")
    print(f"Total not readmitted: {total_not_readmitted:,} ({total_pct_not_readmitted:.2f}%)")
    print()
    
    # Summary table
    print("="*80)
    print("SUMMARY TABLE:")
    print("="*80)
    print()
    print(f"{'File':<20} {'Entries':>10} {'Readmitted':>15} {'Not Readmitted':>18}")
    print("-" * 80)
    for stats in file_stats:
        print(f"{stats['filename']:<20} {stats['entries']:>10,} "
              f"{stats['readmitted']:>8,} ({stats['pct_readmitted']:>5.2f}%) "
              f"{stats['not_readmitted']:>8,} ({stats['pct_not_readmitted']:>5.2f}%)")
    print("-" * 80)
    print(f"{'TOTAL':<20} {total_entries:>10,} "
          f"{total_readmitted:>8,} ({total_pct_readmitted:>5.2f}%) "
          f"{total_not_readmitted:>8,} ({total_pct_not_readmitted:>5.2f}%)")
    print()

if __name__ == "__main__":
    analyze_data()
