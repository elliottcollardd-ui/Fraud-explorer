"""
Train a fraud model on the Kaggle credit-card data and score the test set.

Run:   python3.15 train_model.py creditcard.csv
Makes: scored_test.csv  -> then: python3.15 embed_transactions.py scored_test.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(sys.argv[1])

# 1. Split by TIME, not randomly: train on the earlier 70%, test on the later 30%.
#    That's how a real bank uses a model — trained on the past, scoring the future.
df = df.sort_values("Time").reset_index(drop=True)
cut = int(len(df) * 0.7)
train, test = df.iloc[:cut], df.iloc[cut:]

features = [c for c in df.columns if c != "Class"]
def prep(d):
    X = d[features].copy()
    X["Amount"] = np.log1p(X["Amount"])   # amounts have a long tail; log squashes it
    return X

# 2. Logistic regression. class_weight="balanced" because fraud is ~0.17% of rows —
#    without it the model can score 99.8% accuracy by calling everything normal.
model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
model.fit(prep(train), train["Class"])

# 3. Score the unseen test set and report.
y, p = test["Class"].values, model.predict_proba(prep(test))[:, 1]
print(f"Train: {len(train):,} rows ({train.Class.sum()} fraud)   Test: {len(test):,} rows ({y.sum()} fraud)\n")
print(f"PR-AUC (average precision): {average_precision_score(y, p):.3f}   <- headline metric for rare fraud")
print(f"ROC-AUC:                    {roc_auc_score(y, p):.3f}   <- looks good on imbalanced data almost regardless\n")
print(" threshold  recall  precision  flagged  missed")
for t in [0.5, 0.8, 0.9, 0.95, 0.99]:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"   {t:<6}   {recall_score(y, pred, zero_division=0):.0%}     {precision_score(y, pred, zero_division=0):6.1%}   {tp+fp:7,}   {fn:5}")

out = test.copy()
out["fraud_score"] = p.round(4)
out.to_csv("scored_test.csv", index=False)
print("\nWrote scored_test.csv")
