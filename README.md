# Substance Use Treatment Dropout Prediction with Explainable AI

Predicting treatment dropout from SAMHSA's Treatment Episode Data Set (TEDS-D) using gradient-boosted ensembles, with SHAP-based explainability and a structured error/bias analysis — built as an end-to-end, portfolio-grade ML pipeline.

## Project Summary

Substance use and behavioral health treatment programs lose a meaningful share of patients to early dropout — patients who leave against advice, are terminated, or otherwise fail to complete their planned treatment course. Early dropout is strongly associated with worse downstream outcomes (relapse, re-hospitalization, increased system cost), and treatment programs typically have limited staff time to proactively intervene.

This project builds a model that scores a patient's dropout risk at or near admission, using only information available at that point in time, so that case managers could in principle prioritize outreach (check-ins, transportation support, engagement calls) toward the patients most likely to disengage.

**This is a portfolio/demonstration project**, not a deployed clinical tool. See [Ethical Considerations](#ethical-considerations--limitations) below.

## Dataset

- **Source:** [SAMHSA Treatment Episode Data Set – Discharges (TEDS-D)](https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles)
- **Unit of observation:** treatment discharge episodes (not unique patients — a person discharged twice in a year appears as two episodes)
- **Target variable (engineered):**
  - `1` = Dropout — left against professional advice, terminated by facility, or otherwise did not complete the planned treatment course
  - `0` = Completion — treatment completed as planned
- **Features:** demographics (age, gender, race/ethnicity, marital status), treatment context (setting, referral source, length of stay, prior episodes), substance use patterns (primary/secondary/tertiary substance, frequency, route of administration, age at first use), and co-occurring psychiatric comorbidity flags.

> TEDS-D column names shift slightly across release years. The notebook includes a flexible alias-based column mapper (`COLUMN_ALIASES`) so it can be pointed at most TEDS-D vintages with minimal edits — check the codebook PDF for your chosen year and adjust `DROPOUT_CODES` / `COMPLETION_CODES` if needed.

## Approach

1. **Data quality assessment** — sentinel-value handling (TEDS encodes missing data as `-9`), duplicate removal, missingness analysis, outlier capping (IQR-based).
2. **EDA** — target distribution, demographics, substance use patterns, dropout rate by category, correlation analysis.
3. **Feature engineering** — missing indicators, rare-category bucketing, frequency encoding for high-cardinality fields, interaction features (e.g. age × prior episodes), one-hot encoding via `ColumnTransformer`.
4. **Split** — 70/15/15 train/validation/test, stratified on target.
5. **Class imbalance** — compared no balancing, class weighting, and SMOTE; selected class weighting.
6. **Modeling** — Logistic Regression baseline → Random Forest, XGBoost, LightGBM, CatBoost, all behind identical `sklearn` pipelines.
7. **Tuning** — `RandomizedSearchCV` (stratified CV) per model.
8. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, calibration curve, held-out test-set confirmation.
9. **Explainability** — SHAP global importance, beeswarm, dependence plots, individual waterfall/force plots.
10. **Error analysis** — false positive / false negative characterization, threshold sensitivity, fairness considerations.
11. **Bonus** — example (non-deployed) FastAPI + joblib serving sketch.

## Results

Tuned model comparison on the held-out **test** set (champion model: **XGBoost**):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost (tuned)** | 0.875 | 0.839 | 0.877 | 0.857 | **0.949** | **0.936** |
| LightGBM (tuned) | 0.878 | 0.848 | 0.871 | 0.860 | 0.952* | 0.939* |
| CatBoost (tuned) | 0.872 | 0.837 | 0.873 | 0.855 | 0.948* | 0.934* |
| Random Forest (tuned) | 0.859 | 0.811 | 0.876 | 0.842 | 0.939* | 0.921* |
| Logistic Regression (baseline) | 0.776 | 0.723 | 0.776 | 0.749 | 0.860 | 0.826 |

<sub>*Validation-set scores shown for non-champion models; only the champion was evaluated once on the held-out test set, per standard practice.</sub>

XGBoost and LightGBM were effectively tied (within ~0.001 ROC-AUC); XGBoost was selected as champion by validation ROC-AUC. Test-set performance (0.949 ROC-AUC) closely matched validation (0.952), indicating the tuned model generalizes well rather than overfitting to the validation split.

## Key Drivers of Dropout Risk (SHAP)

Across SHAP global importance and dependence plots, the strongest predictors of dropout risk were consistent with substance use treatment literature:

- **Number of prior treatment episodes** — more prior episodes associated with elevated dropout risk
- **Referral source** — coerced/court-mandated referrals show different risk profiles than self-referrals
- **Treatment setting** — outpatient settings without intensive structure show higher dropout signal than residential settings
- **Co-occurring psychiatric comorbidity flag** — presence of a flagged psychiatric condition associated with elevated risk
- **Age and age at first substance use**

See the notebook's Explainable AI section for full SHAP visualizations and per-patient waterfall/force plot examples.

## Error Analysis Highlights

- False negative rate (missed dropouts) on validation: 5.3% — the costlier error class for this use case, since it represents at-risk patients who wouldn't be flagged for outreach.
- Errors cluster near the 0.5 decision threshold, suggesting genuinely ambiguous cases rather than systematic blind spots — threshold tuning (favoring recall) is a natural next lever without retraining.
- A formal subgroup fairness audit (race, ethnicity, gender, referral source) is flagged as a required next step before any real-world use — not performed in this notebook. See [Ethical Considerations](#ethical-considerations--limitations).

## Repository Contents

```
TEDS_Treatment_Dropout_Prediction_XAI.ipynb   # Full end-to-end notebook
README.md                                     # This file
```

## How to Run

1. Open the notebook in Google Colab.
2. (Recommended) Switch to a GPU runtime: `Runtime → Change runtime type → T4 GPU`.
3. Run the setup cell to install dependencies.
4. Download a TEDS-D CSV from [SAMHSA's data portal](https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles) (choose **TEDS-D**, any year, **Delimited/CSV** format) and upload it when prompted, or set `DATA_PATH` directly.
5. Run all cells top to bottom. `DEV_SAMPLE_SIZE` controls how much data is used during development — set to `None` for a full final run.

## Ethical Considerations & Limitations

- This model is built on **administrative data**, not clinical-grade records. Discharge reason codes (e.g. "terminated by facility") reflect program policy and reporting practices as much as patient behavior — they are not a perfect ground truth for "patient choice to leave."
- The appropriate use case for a model like this is to **prioritize proactive outreach resources**, never to deny, ration, or flag patients punitively.
- A full subgroup fairness audit (differential error rates by race, ethnicity, gender, and referral source) was **not** performed in this notebook and would be required before any real-world application, given well-documented disparities in substance use treatment access and outcomes.
- TEDS is episode-level, not patient-level — repeated episodes for the same individual are treated as independent rows.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `imbalanced-learn` (SMOTE) · `SHAP` · `matplotlib` / `seaborn` · `joblib` · `FastAPI` (deployment sketch only)

---

*Built as a portfolio project demonstrating end-to-end ML engineering: data quality assessment, leakage-free pipelines, imbalanced classification, hyperparameter tuning, explainability, and responsible-AI judgment applied to a real public health dataset.*
