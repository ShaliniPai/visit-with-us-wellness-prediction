# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# CONFIGURATION

HF_DATASET_REPO = "Shalini94/tourism-wellness-dataset"
RAW_DATA_FILE = "tourism.csv"
TARGET_COL = "ProdTaken"

api = HfApi(token=os.getenv("HF_TOKEN"))

# 1. LOAD DATA FROM HF DATASET

# Download dataset file locally
api.hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename=RAW_DATA_FILE,
    local_dir="."
)

df = pd.read_csv(RAW_DATA_FILE)
print("Dataset loaded from Hugging Face successfully")

# 2. DATA CLEANING

# Drop identifier column (not useful for ML)
df.drop(columns=["CustomerID"], inplace=True)

# Standardize Gender values (e.g., 'Fe Male')
df["Gender"] = df["Gender"].str.replace(" ", "").str.capitalize()

# Handle missing values (simple strategy)
df.fillna(0, inplace=True)

# Encode categorical columns
categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("Data cleaning and encoding completed")

# 3. TRAIN–TEST SPLIT
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save locally
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Train-test split completed and files saved locally")

# 4. UPLOAD BACK TO HF DATASET
files_to_upload = [
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv"
]

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"processed/{file}",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

print("Processed datasets uploaded back to Hugging Face successfully")
