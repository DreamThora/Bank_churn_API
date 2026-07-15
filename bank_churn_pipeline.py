"""
Bank churn binary classification — pipeline module.

Expected input columns (raw):

    Required for prediction (values are read by feature-engineering steps):
        CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

    Required but unused (must exist so drop_useless_feature does not error;
    values themselves are never read):
        id, CustomerId, Surname

    Target column (present at training time only, never at inference):
        Exited

Generated from binary-classification-with-a-bank-churn.ipynb via the
productionization skill.

This file is an import target only. To produce the deployable .joblib, run
train_and_save.py — do NOT run this file directly to save. See the skill's
Common Pitfalls entry on pickle's __main__ path binding.

Every feature-engineering function below is copied verbatim from the notebook.
See the summary in chat for what was dropped, what was preserved as-is despite
being unused downstream, and what's still needed before this is deploy-ready.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# --- feature engineering functions ---------------------------------------
# Defined in the same order they're used as steps in churn_pipeline below,
# per Step 3 of the productionization skill — read top to bottom to follow
# the same sequence as the Pipeline() call.

def drop_useless_feature(df):
    return df.drop(['id', 'CustomerId', 'Surname'], axis=1)


def transform_geo_gender(df):
    df['Germany'] = (df['Geography'] == "Germany").astype(int)
    df['InActive'] = np.where((df['IsActiveMember'].astype(int)) == 1, 0, 1)
    df['InActive+Germany'] = df['Germany'] * df['InActive']

    df['Female'] = (df['Gender'] == 'Female').astype(int)
    df = df.drop(['Geography', 'IsActiveMember', 'Gender', 'Tenure', 'HasCrCard'], axis=1)
    return df


def create_InActive_x_HighProduct(df):
    df['HighProduct'] = (df['NumOfProducts'] >= 3).astype(int)
    df['InActive_x_HighProduct'] = df['InActive'] * df['HighProduct']
    return df


def create_Female_x_Germany(df):
    df['Female_x_Germany'] = df['Female'] * df['Germany']
    return df


age_label = ['Student', 'Early Career', 'Adults', 'Pre Retirement', 'Near Retirement', 'Retirees']
age_bins = [-1, 25, 35, 45, 55, 65, np.inf]


def transform_age(df):
    df['Age_Bracket'] = pd.cut(df['Age'], bins=age_bins, labels=age_label)
    df.drop('Age', axis=1, inplace=True)
    return df


def add_bracket_features(df):
    # CreditScore — FICO Standard
    df['CreditScore_bracket'] = pd.cut(
        df['CreditScore'],
        bins=[349, 579, 669, 739, 850],
        labels=['Poor', 'Fair', 'Good', 'Excellent'],
        include_lowest=True
    )

    # Balance
    df['Balance_bracket'] = pd.cut(
        df['Balance'],
        bins=[-1, 0, 120000, np.inf],
        labels=['Zero_Balance', 'Mid_Balance', 'High_Balance'],
        include_lowest=True
    )
    # EstimatedSalary
    df['Salary_bracket'] = pd.cut(
        df['EstimatedSalary'],
        bins=[0, 74835, 155616, np.inf],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    return df


def create_HasBalance_Feature(df):
    df['HasBalance'] = (df['Balance'] > 0).astype(int)
    return df


def drop_CreditScore_Balance_Salary(df):
    col = ['CreditScore', 'Balance', 'EstimatedSalary', 'CreditScore_bracket', 'Balance_bracket', 'Salary_bracket']
    df = df.drop(col, axis=1)
    return df


# --- pipeline assembly -----------------------------------------------------
# Step order matches the function definition order above and the order the
# notebook applied them in. All steps are stateless (Step 2) or sklearn-native
# (OneHotEncoder) — no custom transformer class was needed for this notebook.

churn_pipeline = Pipeline([
    ("drop_useless",       FunctionTransformer(drop_useless_feature)),
    ("geo_gender",         FunctionTransformer(transform_geo_gender)),
    ("inactive_x_product", FunctionTransformer(create_InActive_x_HighProduct)),
    ("female_x_germany",   FunctionTransformer(create_Female_x_Germany)),
    ("age_bracket",        FunctionTransformer(transform_age)),
    ("unused_brackets",    FunctionTransformer(add_bracket_features)),  # kept for fidelity — see note 1 in chat
    ("has_balance",        FunctionTransformer(create_HasBalance_Feature)),
    ("drop_unused",        FunctionTransformer(drop_CreditScore_Balance_Salary)),
    ("onehot",             OneHotEncoder(sparse_output=False, drop='first')),
    ("model",              LogisticRegression(max_iter=500, class_weight='balanced')),
])
