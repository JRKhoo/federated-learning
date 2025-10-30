import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetic_data.csv')

# 80/20 train/test split
df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

# split training data into 3 hospitals
hospital_splits = np.array_split(df_train.sample(frac=1, random_state=1), 3)

hospital_splits[0].to_csv('hospital1.csv', index=False)
hospital_splits[1].to_csv('hospital2.csv', index=False)
hospital_splits[2].to_csv('hospital3.csv', index=False)
df_test.to_csv('test_data.csv', index=False)

print(f"Test set size: {len(df_test)}")
print(f"Hospital 1 size: {len(hospital_splits[0])}")
print(f"Hospital 2 size: {len(hospital_splits[1])}")
print(f"Hospital 3 size: {len(hospital_splits[2])}")