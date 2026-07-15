"""
Train and save the bank churn pipeline.

Run this file (`python train_and_save.py`) to produce
churn_model_pipeline.joblib — the artifact a downstream API will load.

Do NOT run bank_churn_pipeline.py directly to save. Because this file imports
the pipeline module, pickle records each feature-engineering function's module
path as `bank_churn_pipeline`, so joblib.load(...) from another file can find
them again. Running the pipeline module directly would bind those paths to
`__main__` instead, causing an AttributeError at load time. See the
productionization skill's Common Pitfalls for details.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# The import below is what makes pickle record the correct module path
# for the feature-engineering functions used inside churn_pipeline.
from bank_churn_pipeline import churn_pipeline


if __name__ == "__main__":
    df = pd.read_csv("train.csv")  # original notebook read from a Kaggle path — see note in chat

    x_train, x_valid = train_test_split(df, test_size=0.2, random_state=42)
    y_train = x_train.pop("Exited")
    y_valid = x_valid.pop("Exited")

    churn_pipeline.fit(x_train, y_train)

    y_valid_prob = churn_pipeline.predict_proba(x_valid)[:, 1]
    print("Validation ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))

    joblib.dump(churn_pipeline, "churn_model_pipeline.joblib")
    print("Saved churn_model_pipeline.joblib")
