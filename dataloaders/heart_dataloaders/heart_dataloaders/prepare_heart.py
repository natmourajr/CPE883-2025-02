from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# from UCI documentation
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

OUTPUT_DIR = "../../../datasets/heart_disease/"

# replace missing '?' with NaN
data = pd.read_csv(
    OUTPUT_DIR + "processed.cleveland.data",
    header=None,
    names=COLUMNS,
    na_values="?"
)

# drop rows with missing values
data = data.dropna(axis=0)


# Convert all object (string) type columns to numeric using LabelEncoder
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "full"), exist_ok=True)

# target: 0 = no disease, 1 = disease
data["target"] = (data["target"] > 0).astype(int)

train, test = train_test_split(data, test_size=0.2, random_state=42)
data.to_csv(OUTPUT_DIR+"full/heart_disease.csv", index=False)
train.to_csv(OUTPUT_DIR+"train/heart_disease.csv", index=False)
test.to_csv(OUTPUT_DIR+"test/heart_disease.csv", index=False)

print(f"Train set: {len(train)} samples")
print(f"Test set: {len(test)} samples")
print(f"Full set: {len(data)} samples")
