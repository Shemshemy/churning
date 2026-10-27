# iGaming Player Churn — CRISP-DM Case Study

An end-to-end portfolio project exploring how early player activity can help
identify elevated churn risk and inform a responsible retention pilot. The work
follows the six CRISP-DM phases and keeps data preparation, feature engineering,
and modeling logic in reusable Python modules.

## Executive summary

The business question is whether player activity in an initial 30-day
observation window can help rank players who have no qualifying activity in the
following 45 days. The prepared cohort contains **5,028 eligible players**, with
a **57.9% churn rate** in the saved project report.

The notebook compares baseline classifiers, then tunes LightGBM with
stratified five-fold cross-validation. In the saved report, the tuned model
achieved a **0.7964 held-out PR-AUC** and a **0.7637 held-out ROC-AUC**. A
scenario analysis estimates retention campaign economics using stated
assumptions of $150 player value, $20 offer cost, and a 25% campaign success
rate. Those are planning assumptions, not causal results; a controlled pilot
would be needed before operational use.

Any real player-retention workflow should include privacy controls, performance
monitoring, and responsible-gambling safeguards.

## CRISP-DM workflow

1. **Business understanding** — define the 30-day observation period, 45-day
   inactivity outcome, and business decision.
2. **Data understanding** — profile the five source tables, time coverage,
   missing values, and player joins.
3. **Data preparation** — normalize types in DuckDB, create complete
   player-relative windows, and aggregate leakage-controlled features.
4. **Modeling** — compare baseline models and tune LightGBM by cross-validated
   average precision (PR-AUC).
5. **Evaluation** — compare the selected configuration with the baseline on the
   untouched holdout.
6. **Deployment** — estimate threshold-based campaign scenarios and outline a
   controlled pilot.

## Repository map

```text
data/        Data setup instructions and locally supplied raw CSVs
notebooks/   Complete CRISP-DM walkthrough
src/         Reusable cleaning, feature, and model functions
artifacts/   Prepared Parquet dataset and serialized model payload
reports/     Saved project summary
```

## Getting started

Use Python 3.10 or newer. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Obtain the authorized raw source exports and place them in `data/raw/` using
the filenames in [data/README.md](data/README.md). Raw CSVs are git-ignored.
Open [notebooks/iGaming_Churn_CRISP_DM.ipynb](notebooks/iGaming_Churn_CRISP_DM.ipynb)
and run all cells from the top. The notebook writes the prepared dataset,
serialized model, and report to `artifacts/` and `reports/`.

## Reproducibility and model use

The model payload includes the fitted estimator, ordered feature names, and a
threshold selected by the notebook's business-scenario analysis. Load it with
`joblib.load`; use only the recorded feature list in the same order used for
training. The notebook reruns the cross-validation search and can take several
minutes depending on available CPU resources.

The committed dataset and model payload are reproducibility artifacts, not
substitutes for the excluded raw inputs. Rebuild them by running the complete
pipeline against the authorized source data.

## Limitations

- The churn label is an operational definition based on observed activity, not
  confirmed account closure.
- The target includes cash-game, tournament, and deposit activity; model inputs
  currently focus on cash-game and deposit aggregates.
- The campaign return calculation assumes a fixed intervention success rate.
  It does not establish incremental causal impact.
- This is a case study, not a production decision system. Any real use needs
  governance, monitoring, privacy controls, and responsible-gambling review.
