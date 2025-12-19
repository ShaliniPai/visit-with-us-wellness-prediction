from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HF_DATASET_REPO = "Shalini94/tourism-wellness-dataset"
RAW_DATA_FILE = "tourism.csv"
TARGET_COL = "ProdTaken"

api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Ensure dataset repo exists
# -----------------------------
try:
    api.repo_info(repo_id=HF_DATASET_REPO, repo_type="dataset")
    print("Dataset repo exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        private=False
    )
    print("Dataset repo created.")

# -----------------------------
# Download raw dataset
# -----------------------------
api.hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename=RAW_DATA_FILE,
    local_dir="."
)

df = pd.read_csv(RAW_DATA_FILE)
print("Dataset loaded successfully")

# -----------------------------
# Data cleaning
# -----------------------------
df.drop(columns=["CustomerID"], inplace=True)
df["Gender"] = df["Gender"].str.replace(" ", "").str.capitalize()
df.fillna(0, inplace=True)

categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# Train-test split
# -----------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# -----------------------------
# Upload processed files
# -----------------------------
for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"processed/{file}",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

print("Processed datasets uploaded successfully")
