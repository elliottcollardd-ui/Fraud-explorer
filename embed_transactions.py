"""
Turn a bank-transactions CSV into 3D points for the Fraud Explorer.

Each transaction becomes a point; transactions that *behave* alike (amount, time of day,
category, distance, velocity...) land near each other. Fraud that sits inside a normal
cluster is the fraud your features can't see.

Install:  pip install pandas scikit-learn umap-learn
Run:      python embed_transactions.py transactions.csv
Options:  --label is_fraud      column with 1 = confirmed fraud      (auto-detected if omitted)
          --score fraud_score   column with model risk score 0-1     (optional; if missing, a
                                neighbour-based score is computed from the fraud label)
          --group category      column to colour by in "Group" mode  (optional)
          --sample 5000         max normal transactions kept (all fraud is always kept)
          --drop col1,col2      columns to ignore (IDs, names, card numbers...)
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default="transactions_points.json")
ap.add_argument("--label")
ap.add_argument("--score")
ap.add_argument("--group")
ap.add_argument("--sample", type=int, default=5000)
ap.add_argument("--drop", default="")
a = ap.parse_args()

df = pd.read_csv(a.csv)
drop = {c.strip() for c in a.drop.split(",") if c.strip()}
cols = {c.lower(): c for c in df.columns if c not in drop}
def find(given, names):
    if given:
        return given
    return next((cols[n] for n in names if n in cols), None)

label = find(a.label, ["is_fraud", "class", "fraud", "isfraud", "label"])
score = find(a.score, ["fraud_score", "score", "risk_score", "probability", "fraud_probability"])
group = find(a.group, ["category", "merchant_category", "type", "transaction_type"])
print(f"label={label}  score={score}  group={group}")

# Fraud is rare: keep every fraud row, sample the rest so UMAP stays fast.
if label:
    fraud = df[df[label] == 1]
    normal = df[df[label] != 1]
    if len(normal) > a.sample:
        normal = normal.sample(a.sample, random_state=42)
    df = pd.concat([fraud, normal]).reset_index(drop=True)
elif len(df) > a.sample:
    df = df.sample(a.sample, random_state=42).reset_index(drop=True)
print(f"{len(df)} transactions kept")

# ---- build the feature matrix ----
skip = {label, score} | drop
feats = []
for c in df.columns:
    if c in skip:
        continue
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float).fillna(s.median())
        if v.min() >= 0 and v.skew() > 2:          # money-like columns: compress the long tail
            v = np.log1p(v)
        feats.append(v.rename(c))
        continue
    t = pd.to_datetime(s, errors="coerce", format="mixed")
    if t.notna().mean() > 0.9:                       # timestamps -> hour and weekday
        h = t.dt.hour.fillna(0) + t.dt.minute.fillna(0) / 60
        feats += [np.sin(2 * np.pi * h / 24).rename(c + "_hsin"), np.cos(2 * np.pi * h / 24).rename(c + "_hcos"),
                  t.dt.dayofweek.fillna(0).rename(c + "_dow")]
    elif s.nunique() <= 15:                          # small categoricals -> one-hot
        feats.append(pd.get_dummies(s.astype(str), prefix=c).astype(float))
    # else: high-cardinality text (names, merchant strings, IDs) is display-only

X = StandardScaler().fit_transform(pd.concat(feats, axis=1).values)
print(f"{X.shape[1]} feature columns")

# No model score? Use "how fraud-like is the neighbourhood": share of the 10 most similar
# transactions (in full feature space) that are confirmed fraud.
score_source = "column: " + score if score else None
if not score and label:
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)
    is_f = (df[label] == 1).values
    df["_nbr_score"] = is_f[idx[:, 1:]].mean(axis=1).round(3)
    score, score_source = "_nbr_score", "share of 10 most similar transactions that are fraud"

coords = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.5, random_state=42).fit_transform(X)

# ---- write points ----
show = [c for c in df.columns if c not in drop and not re.fullmatch(r"V\d+", c)][:12]
def clean(v):
    if isinstance(v, (bool, np.bool_)): return int(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (np.floating, float)): return None if np.isnan(v) else round(float(v), 4)
    return str(v)

points = []
for (x, y, z), (_, r) in zip(coords, df.iterrows()):
    points.append({
        "x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4),
        "fraud": int(r[label] == 1) if label else None,
        "score": clean(r[score]) if score else None,
        "group": str(r[group]) if group else None,
        "fields": {c: clean(r[c]) for c in show},
    })
with open(a.out, "w") as f:
    json.dump({"score_source": score_source, "points": points}, f)
print(f"Wrote {len(points)} points to {a.out}")
