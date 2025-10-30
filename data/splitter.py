import pandas as pd
import numpy as np

file_path = 'diabetic_data.csv'
data = pd.read_csv(file_path)

# shuffle data, reducing any ordering bias
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# large dataset, use 80/20 split for train/test
test_size = int(len(data) * 0.2)  # approx 20% of entire data set
train_size = (len(data) - test_size) // 3  # split remaining data into 3 parts

test_set = data.iloc[:test_size]
train_set_1 = data.iloc[test_size:test_size + train_size]
train_set_2 = data.iloc[test_size + train_size:test_size + 2 * train_size]
train_set_3 = data.iloc[test_size + 2 * train_size:test_size + 3 * train_size]

test_set.to_csv('test_set.csv', index=False)
train_set_1.to_csv('train_set_1.csv', index=False)
train_set_2.to_csv('train_set_2.csv', index=False)
train_set_3.to_csv('train_set_3.csv', index=False)

print("Data split successfully.")