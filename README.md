# Fraud Explorer

A fraud-detection baseline on 284,807 card transactions, plus an interactive 3D view for inspecting the fraud the model misses.

![Fraud Explorer showing caught and missed fraud](screenshot.png)

## The problem

A fraud model's headline metrics don't tell you *which* fraud it misses or why. This project trains a classifier, then plots every test transaction in 3D so that transactions which behave alike sit together. Missed fraud can then be inspected directly: is it near other fraud, or hidden among normal spending?

## Data

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions, 492 fraudulent (0.17%). Features `V1`–`V28` are anonymised PCA components; `Time` and `Amount` are raw. The CSV is not included in this repo.

## Results

Logistic regression, trained on the earliest 70% of transactions (199,364 rows, 384 fraud) and tested on the latest 30% (85,443 rows, 108 fraud).

- **PR-AUC: 0.751**
- ROC-AUC: 0.979

| Threshold | Recall | Precision | Flagged | Fraud missed |
|-----------|--------|-----------|---------|--------------|
| 0.50      | 88%    | 5.4%      | 1,768   | 13           |
| 0.80      | 86%    | 17.3%     | 539     | 15           |
| 0.90      | 84%    | 27.2%     | 335     | 17           |
| 0.95      | 82%    | 35.6%     | 250     | 19           |
| 0.99      | 81%    | 49.7%     | 175     | 21           |

## What I found

- **ROC-AUC is misleading here.** It reads 0.979 while only 5.4% of flags at the default threshold are real fraud. With 0.17% fraud, PR-AUC and the precision/recall table are the honest measures.
- **Caught fraud clusters; missed fraud doesn't.** At a 0.95 threshold, the 89 frauds the model catches form one tight group. The 19 it misses are scattered individually among normal transactions, so they look legitimate on these features and a threshold change alone won't recover them.

TODO (Elliott): rotate the view and confirm the second point holds from several angles. Add anything you found when you clicked the three missed frauds that sit together.

## Why I made these choices

TODO (Elliott): write each of these in your own words, two or three sentences each. These are the interview questions.

- Why split by time instead of randomly?
- Why `class_weight="balanced"`?
- Why log-transform `Amount`?
- Why PR-AUC over accuracy or ROC-AUC?
- Why keep all fraud but only sample 5,000 normal transactions for the 3D view?
- What does UMAP preserve, and what does it distort?

## Run it

```
pip install pandas scikit-learn umap-learn
python3 train_model.py creditcard.csv          # trains, prints the table, writes scored_test.csv
python3 embed_transactions.py scored_test.csv  # writes transactions_points.json
```

Open `fraud-explorer.html` in a browser and drop `transactions_points.json` onto it.

## Limitations

- **The 3D view overstates precision.** It shows all test fraud but only 5,000 of the 85,335 normal test transactions, so false alarms are under-represented. The table above is measured on the full test set.
- **UMAP distorts distances.** Clusters are a guide to what to inspect, not proof.
- **One model, one split.** Logistic regression is a baseline; there is no cross-validation or comparison with tree-based models yet.
- **Anonymised features.** `V1`–`V28` can't be interpreted, so the view shows *where* missed fraud sits but not *why* in business terms.

## Built with

Python, pandas, scikit-learn, UMAP. The code was written with AI assistance (Claude); I ran the experiments and wrote the analysis.
