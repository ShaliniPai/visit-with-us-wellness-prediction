
# ===============================
# Model Training with Experiment Tracking
# ===============================

# --------- Imports ----------
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ===============================
# MLflow Configuration
# ===============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-model-training")

# ===============================
# Hugging Face Configuration
# ===============================
HF_DATASET_REPO = "Shalini94/tourism-wellness-dataset"
HF_MODEL_REPO = "Shalini94/tourism-model"

api = HfApi()

# ===============================
# Load Train / Test Data from HF
# ===============================
X_train = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/X_train.csv"
)
X_test = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/X_test.csv"
)
y_train = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/y_train.csv"
).squeeze()
y_test = pd.read_csv(
    f"hf://datasets/{HF_DATASET_REPO}/y_test.csv"
).squeeze()

print("Dataset loaded successfully.")

# ===============================
# Feature Groups
# ===============================
numeric_features = [
    "Age",
    "DurationOfPitch",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# ===============================
# Preprocessing Pipeline
# ===============================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ===============================
# Model Definition
# ===============================
rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# ===============================
# Hyperparameter Grid
# ===============================
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 5],
    "randomforestclassifier__min_samples_leaf": [1, 2]
}

# ===============================
# Full Pipeline
# ===============================
pipeline = make_pipeline(
    preprocessor,
    rf_model
)

# ===============================
# Training + Tracking
# ===============================
with mlflow.start_run():

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # ---------------------------
    # Log all tuned parameters
    # ---------------------------
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # ---------------------------
    # Evaluation
    # ---------------------------
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    train_f1 = f1_score(y_train, train_preds)
    test_f1 = f1_score(y_test, test_preds)

    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1
    })

    # ---------------------------
    # Save Model
    # ---------------------------
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/tourism_best_model.joblib"
    joblib.dump(best_model, model_path)

    # ---------------------------
    # Log Model Artifact
    # ---------------------------
    mlflow.log_artifact(model_path, artifact_path="model")

    # ---------------------------
    # Register Model to HF Hub
    # ---------------------------
    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        print("Model repo exists. Using existing repo.")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            private=False
        )
        print("Model repo created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="tourism_best_model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

    print("Model training, tracking, and registration completed successfully.")
