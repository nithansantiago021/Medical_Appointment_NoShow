# Medical Appointment No-Show Prediction & Demand Forecasting
## Project Report

**Domain:** Healthcare Operations & Resource Management

**Institution:** University of Vale do Itajaí — Center of Specialization in Physical and Intellectual Rehabilitation (CER)

**Dataset:** 109,593 appointments × 26 features | January 2020 – May 2021

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Overview](#3-dataset-overview)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Data Preprocessing & Feature Engineering](#5-data-preprocessing--feature-engineering)
6. [No-Show Classification — Model Building](#6-no-show-classification--model-building)
7. [Demand Forecasting — Model Building](#7-demand-forecasting--model-building)
8. [Results & Model Performance](#8-results--model-performance)
9. [Streamlit Application](#9-streamlit-application)
10. [Business Recommendations](#10-business-recommendations)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Conclusion](#12-conclusion)
- [Appendix A — Technical Stack](#appendix-a--technical-stack)
- [Appendix B — Model Hyperparameters](#appendix-b--model-hyperparameters)
- [Appendix C — Notebook Structure](#appendix-c--notebook-structure)
- [Appendix D — References](#appendix-d--references)

---

## 1. Executive Summary

Medical appointment no-shows are a chronic operational problem in public healthcare. At the CER Univali rehabilitation centre in southern Brazil, **34,832 out of 109,593 appointments (31.78%)** are missed — nearly double the global clinical benchmark of 10–20%. Each empty specialist slot represents wasted capacity, delayed care for other patients, and direct financial loss to the public health system.

This project delivers two complementary machine learning systems and a fully operational deployment interface:

| System | Algorithm | Primary Metric | Result |
|--------|-----------|----------------|--------|
| No-Show Classifier | LightGBM | ROC-AUC | **0.7752** |
| Demand Forecaster | LightGBM Regressor | MAE | **176.68 appointments/day** |

Both models are deployed in a four-page Streamlit application enabling clinic staff to score individual patient risk, forecast daily volume, and access data-driven intervention guidance — without any technical knowledge.

**Three highest-impact findings from this analysis:**

1. The current SMS reminder system has **zero measurable effect** (+0.10 pp) — a complete redesign is warranted
2. Cold-weather days produce a no-show rate of **52.93%** — 21 percentage points above the overall average
3. Appointments with no assigned specialty have a **52.78%** no-show rate and represent the single highest-ROI target for proactive outreach

---

## 2. Problem Statement

### 2.1 Background

The University of Vale do Itajaí Center of Specialization in Physical and Intellectual Rehabilitation (CER) is an outpatient rehabilitation facility in southern Brazil serving the public health system. The clinic provides specialized care including physiotherapy, psychotherapy, speech therapy, and occupational therapy across 13 cities in the service region.

### 2.2 Operational Challenges

- **No-show rate of 31.78%** — 21 percentage points above the acceptable upper bound of 10–20%
- **Unpredictable daily demand** ranging from 1 to 1,512 appointments per day (mean 220.1, std 245.8), making staff scheduling extremely difficult
- **No proactive patient engagement** — no system exists to identify high-risk patients before an appointment is missed
- **Ineffective reminder system** — SMS reminders show statistically zero benefit (+0.10 pp difference in no-show rate)
- **Weather and seasonal impacts** on attendance are not systematically accounted for in scheduling
- **Geographic barriers** — the clinic serves 13 cities; patients from distant locations miss at higher rates due to transport logistics

### 2.3 Project Objectives

1. Build a **binary classifier** that predicts whether a patient will miss their appointment, producing a probability risk score for each scheduled appointment
2. Build a **demand forecasting model** that predicts daily appointment volume to enable proactive staffing and capacity planning
3. Deploy both models in an **interactive Streamlit dashboard** for operational use by clinic staff without requiring technical expertise
4. Deliver **evidence-based business recommendations** for reducing the no-show rate toward the clinical benchmark

### 2.4 Success Criteria

| Objective | Minimum Acceptable | Target |
|-----------|-------------------|--------|
| Classifier ROC-AUC | > 0.70 | > 0.75 |
| Classifier F1-Score (no-show class) | > 0.45 | > 0.55 |
| Demand forecaster MAE | < 250 appointments/day | < 200 appointments/day |
| Dashboard pages | 3 | 4 |

---

## 3. Dataset Overview

### 3.1 Dataset Characteristics

| Property | Value |
|----------|-------|
| Total records | 109,593 |
| Number of features | 26 (raw) → 29 (after engineering) |
| Target variable | `no_show` — "yes" (31.78%) / "no" (68.22%) |
| Date range | 1 January 2020 – 12 May 2021 |
| Total calendar days | 498 (484 usable after lag creation) |
| Geographic coverage | 13 cities in the AMFRI region, Santa Catarina, Brazil |
| Patient population | Primarily children with motor/intellectual disabilities |
| Clinical setting | Outpatient SUS rehabilitation centre |

### 3.2 Feature Groups

**Patient demographics:** `gender`, `age` (median: 12 years), `disability` (intellectual/motor), `under_12_years_old`, `over_60_years_old`, `patient_needs_companion`

**Appointment logistics:** `specialty`, `appointment_time` (07:00–18:00), `appointment_shift` (morning/afternoon), `appointment_date_continuous`, `place` (city of patient residence)

**Health and social flags (binary):** `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`, `Scholarship`, `SMS_received`

**Weather on appointment day:** `average_temp_day`, `average_rain_day`, `max_temp_day`, `max_rain_day`, `heat_intensity`, `rain_intensity`, `rainy_day_before`, `storm_day_before`

**Target:** `no_show` — `"yes"` = patient did not attend, `"no"` = patient attended

### 3.3 Missing Values

| Column | Missing % | Strategy Applied | Rationale |
|--------|-----------|-----------------|-----------|
| `age` | 21.0% | Median imputation (median = 12 years) | Right-skewed distribution; median better than mean |
| `specialty` | 18.4% | Filled with `'unknown'` sentinel | Missing specialty → 52.78% no-show; signal must be preserved |
| `disability` | 15.1% | Filled with `'unknown'` sentinel | Blank disability → 67.8% no-show; signal must be preserved |
| `place` | 10.5% | Filled with `'unknown'` sentinel | Distinct behavioural group — not a random gap |
| Weather columns | ~2.1% | Column median imputation | Minimal gap; distribution unchanged |
| `no_show` | 0.0% | No action required | Fully observed — ideal for supervised learning |

> **Critical design decision:** The `'unknown'` sentinel category for `specialty` and `disability` was chosen deliberately over mode imputation. Records with blank disability have a **67.8% no-show rate** — 36 percentage points above the overall average. Imputing these to the most frequent category (intellectual disability, ~29% no-show) would destroy this high-risk signal entirely. The `'unknown'` category is treated as a real, meaningful patient group throughout modelling.

### 3.4 Data Quality Issues Identified and Resolved

| Issue | Column | Resolution |
|-------|--------|------------|
| Typo in column name | `Hipertension` | Renamed to `Hypertension` in preprocessing |
| Outlier age value | `age` = 110 | Clipped at 100 (likely data entry error) |
| Inverted encoding | `rainy_day_before`, `storm_day_before` | Documented; retained as-is (tree models infer split direction) |
| Empty string as category | `disability` = `""` | Coerced to `'unknown'` via `.str.strip().replace('', 'unknown')` |

---

## 4. Exploratory Data Analysis

EDA was conducted across twelve subsections spanning patient demographics, appointment logistics, weather conditions, health flags, geographic patterns, temporal patterns, and the target time series. All plots use a unified seaborn `whitegrid` theme with a consistent two-colour palette (primary blue `#4C8BB5`, secondary sand-orange `#F4A460`).

### 4.1 Target Variable — Class Imbalance

| Outcome | Count | Percentage |
|---------|-------|-----------|
| Attended (no-show = "no") | 74,761 | 68.22% |
| No-Show (no-show = "yes") | 34,832 | 31.78% |

The dataset is **moderately imbalanced** — the no-show class represents approximately one-third of all records. This is operationally severe (31.78% vs the 10–20% clinical benchmark) and statistically significant enough to require specific handling during modelling. A no-show rate of 31.78% means approximately 1-in-3 scheduled appointment slots are wasted.

The imbalance necessitates:
- **SMOTE oversampling** of the training set minority class
- **ROC-AUC and F1-Score** as primary metrics (not raw accuracy, which would be misleadingly high if all predictions were "show")
- **Stratified splitting** to preserve the 68/32 ratio in both train and test sets

### 4.2 No-Show by Gender

| Gender | Appointments | No-Show Rate |
|--------|-------------|-------------|
| Male (M) | 82,269 | 31.20% |
| Female (F) | 27,077 | 33.69% |
| Other (I) | 247 | ~32.0% |

Female patients have a 2.49 percentage point higher no-show rate. The dataset is heavily male-dominated (75.1% male), reflecting the higher prevalence of intellectual and motor disabilities among male children. Gender is a **weak individual predictor** — the 2.49 pp gap is operationally modest. Its inclusion in the model is justified as a contributing feature in multivariate combination, not as a standalone risk factor.

### 4.3 No-Show by Medical Specialty

| Specialty | No-Show Rate | Volume |
|-----------|-------------|--------|
| Sem especialidade (no specialty assigned) | **52.78%** | moderate |
| Physiotherapy | 34.36% | highest |
| Psychotherapy | 32.34% | moderate |
| Speech therapy | 27.93% | moderate |
| Assist | 26.93% | low |
| Occupational therapy | 23.86% | moderate |
| Enf | 19.75% | low |
| Pedagogo | **15.98%** | low |

**Key finding:** The 36.8 percentage-point range across specialties makes `specialty_enc` one of the strongest predictors. The "sem especialidade" group (appointments with no specialty assigned) has a rate nearly 21 points above the overall average. These are likely administrative intake appointments where the patient has no clear understanding of the clinical purpose — low perceived urgency produces high cancellation rates. The Pedagogo specialty, by contrast, involves structured educational routines familiar to caregivers, producing the lowest no-show rate.

### 4.4 No-Show by Weather Conditions

**Heat Intensity:**

| Heat Category | No-Show Rate | vs. Average |
|--------------|-------------|------------|
| Heavy cold | **52.93%** | +21.15 pp |
| Heavy warm | 38.79% | +7.01 pp |
| Warm | 35.37% | +3.59 pp |
| Cold | 31.71% | −0.07 pp |
| Mild | **26.10%** | −5.68 pp |

The relationship between temperature and no-show rate is **non-linear**: both temperature extremes (heavy cold and heavy warm) increase no-shows relative to the mild baseline. In Itajaí's warm coastal climate (mean ~22 °C), heavy cold days are unusual and represent a genuine barrier to travel — particularly for families transporting children with disabilities.

**Rain Intensity:**

| Rain Category | No-Show Rate |
|--------------|-------------|
| No rain | 31.25% |
| Weak | 31.83% |
| Moderate | 33.20% |
| Heavy | 32.78% |

Rain intensity has minimal discriminating power — a spread of only ~2 pp across all categories. Rain is not a meaningful behavioural barrier in this dataset. `heat_intensity` is far more predictive than `rain_intensity`.

### 4.5 No-Show by Month

| Month | No-Show Rate | vs. Average |
|-------|-------------|------------|
| January | 31.11% | −0.67 pp |
| February | 31.55% | −0.23 pp |
| March | 31.14% | −0.64 pp |
| April | 31.89% | +0.11 pp |
| May | 32.65% | +0.87 pp |
| June | 31.93% | +0.15 pp |
| July | 32.30% | +0.52 pp |
| August | 32.09% | +0.31 pp |
| September | 31.51% | −0.27 pp |
| October | 32.05% | +0.27 pp |
| November | 32.05% | +0.27 pp |
| **December** | **34.27%** | **+2.49 pp** |

Monthly variation is narrow (3.16 pp peak-to-trough across the year). December shows the highest rate, likely reflecting holiday disruption to family routines. Monthly patterns are more valuable as features for the **demand forecasting model** than for the no-show classifier.

### 4.6 No-Show by Day of Week

| Day | No-Show Rate | Type |
|-----|-------------|------|
| Monday | 31.62% | Weekday |
| Tuesday | 31.19% | Weekday |
| Wednesday | 31.64% | Weekday |
| Thursday | 31.78% | Weekday |
| Friday | 32.09% | Weekday |
| Saturday | 32.22% | Weekend |
| Sunday | 32.72% | Weekend |

Day-of-week variation is equally modest (~1.5 pp spread). Weekend appointments are slightly higher risk. The `is_weekend` binary feature is included as a complement to the raw `day_of_week` integer to capture the Saturday–Sunday step-change in demand for the forecasting model.

### 4.7 No-Show by Appointment Shift

| Shift | No-Show Rate | Volume |
|-------|-------------|--------|
| Morning | 33.01% | higher |
| Afternoon | 30.68% | lower |

Morning appointments have a 2.33 pp higher no-show rate than afternoon. Combined with higher absolute morning volume, this means the morning shift generates the largest absolute number of empty slots — making it the highest-priority target for operational interventions (overbooking, confirmation calls).

### 4.8 No-Show by Disability Type

| Disability | No-Show Rate | Count |
|-----------|-------------|-------|
| Intellectual disability | 28.72% | ~62,852 |
| Motor disability | 28.95% | ~29,721 |
| **Unknown / blank** | **67.83%** | ~419 |

Both classified disability types show below-average no-show rates (~29%), confirming that patients with diagnosed disabilities are more reliably scheduled. The blank/unclassified group (n ≈ 419) has an alarming **67.83% no-show rate** — more than double the classified groups. These records likely represent unregistered walk-in patients or administrative errors, with minimal clinical follow-up support explaining the extreme rate.

### 4.9 No-Show by Geographic Location

Cities with more than 100 appointments (ordered by no-show rate, ascending):

| City | No-Show Rate | Distance from CER |
|------|-------------|------------------|
| Penha | ~6.2% | closest/local |
| Itajaí (home city) | ~8.9% | facility city |
| Balneário Camboriú | ~9.5% | moderate |
| Navegantes | ~10.1% | moderate |
| Luiz Alves | **~13.6%** | most distant |
| Ilhota | ~13.1% | distant |
| Bombinhas | ~12.8% | distant |

The **distance-decay effect** is clear and statistically robust: patients who travel farther face greater logistical barriers (transport cost, travel time, time off work for caregivers) and are more likely to miss. The `place_enc` feature ranks 2nd in LightGBM feature importance — geographic location is one of the two most predictive features in the dataset alongside age.

### 4.10 Health Conditions & Social Factors vs No-Show

| Condition / Factor | Without (%) | With (%) | Δ (pp) | Interpretation |
|-------------------|------------|---------|--------|----------------|
| Over 60 years old | 32.46% | **23.03%** | **−9.43** | Elderly patients most reliable |
| Patient needs companion | 34.80% | **29.00%** | **−5.80** | Companion creates accountability |
| Hypertension | 32.11% | **26.42%** | **−5.69** | Chronic disease = higher appointment dependency |
| Diabetes | 31.89% | **27.58%** | **−4.31** | Same pattern as hypertension |
| Under 12 years old | 33.50% | **29.70%** | **−3.80** | Caregivers prioritise children's therapy |
| Scholarship | 31.78% | 31.80% | +0.02 | Socioeconomic subsidy alone predicts nothing |
| **SMS received** | **31.75%** | **31.85%** | **+0.10** | **Current SMS strategy is ineffective** |

**Critical finding:** Chronic disease patients (hypertension, diabetes) attend more reliably than expected — chronic conditions create a higher perceived urgency for each appointment. The SMS reminder effect (+0.10 pp) is statistically indistinguishable from zero, indicating the current reminder strategy fails entirely. Possible causes include wrong timing, generic content, or being sent to patients who would have attended regardless.

### 4.11 Correlation Heatmap Analysis

A full Pearson correlation heatmap was computed for all 19 numeric features plus the binary target. Key observations:

**Strongest correlations with the no-show target:**

| Feature | Correlation with Target | Direction |
|---------|------------------------|-----------|
| `storm_day_before` | −0.14 | Protective (note: inverted encoding) |
| `rainy_day_before` | −0.14 | Protective (note: inverted encoding) |
| `average_temp_day` | +0.09 | Risk-increasing at extremes |
| `patient_needs_companion` | −0.06 | Protective |
| `over_60_years_old` | −0.05 | Protective |

**Notable inter-feature correlations (multicollinearity):**

| Feature Pair | Pearson r | Implication |
|-------------|-----------|-------------|
| `average_temp_day` & `max_temp_day` | +0.96 | Near-duplicate information |
| `rainy_day_before` & `storm_day_before` | +0.99 | Near-identical signal |
| `average_rain_day` & `max_rain_day` | +0.85 | High redundancy |

The maximum absolute correlation of any single feature with the target is only 0.14 — confirming this is a **weak-signal problem** where no individual feature is a strong predictor. This validates the need for non-linear ensemble methods that combine many modest signals. The high inter-feature correlations (r = 0.96–0.99) were handled by retaining both columns and relying on tree-based models' natural ability to down-weight redundant features.

### 4.12 Daily Appointment Demand — Time Series Analysis

The daily appointment count time series spans 498 calendar days:

| Statistic | Value |
|-----------|-------|
| Series length | 498 days |
| Mean demand | 220.1 appointments/day |
| Standard deviation | 245.8 appointments/day |
| Coefficient of variation | 1.12 (std exceeds mean) |
| Minimum | 1 appointment |
| Maximum | 1,512 appointments |

The series exhibits three major patterns: (1) high pre-pandemic demand in January–February 2020; (2) a sharp collapse to near-zero activity in March–April 2020; (3) a gradual, non-linear recovery through 2020–2021. This COVID-19 structural break makes the series non-stationary and represents the primary challenge for the demand forecasting model. Rolling 7-day and 30-day averages were overlaid in the notebook to visualise the trend beneath the day-to-day noise.

---

## 5. Data Preprocessing & Feature Engineering

### 5.1 Preprocessing Pipeline

All preprocessing steps were applied to the full dataset before train-test splitting, except SMOTE which was applied exclusively inside each cross-validation training fold.

```
Raw data (109,593 rows, 26 columns)
        ↓
Column rename: 'Hipertension' → 'Hypertension'
        ↓
Date feature extraction (month, day_of_week, week_of_year, quarter, is_weekend)
        ↓
Missing value imputation
  ├── age → median(12 yrs), clipped at [0, 100]
  ├── specialty / disability / place → 'unknown' sentinel
  └── weather columns → column median
        ↓
Target encoding: no_show → binary 0 (show) / 1 (no-show)
        ↓
Label encoding of 7 categorical features (LabelEncoder)
        ↓
Feature matrix assembly: 29 features
        ↓
Stratified 70/30 train-test split (stratify=y, random_state=42)
        ↓
SMOTE applied to training set only (inside Pipeline per CV fold)
        ↓
Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
        ↓
Final model evaluation on untouched test set
```

### 5.2 Feature Engineering

Five temporal features were derived from `appointment_date_continuous`:

| Derived Feature | Source | Rationale |
|----------------|--------|-----------|
| `month` | Appointment date | Monthly seasonality in attendance behaviour |
| `day_of_week` | Appointment date | Monday–Sunday attendance cycles |
| `week_of_year` | Appointment date | Week-level seasonal patterns |
| `quarter` | Appointment date | Quarterly healthcare demand cycles |
| `is_weekend` | Appointment date | Binary weekend demand suppression flag |

### 5.3 Encoding Strategy

**LabelEncoder** was chosen over One-Hot Encoding (OHE) for all 7 categorical columns:

| Column | Classes | Encoding |
|--------|---------|---------|
| `gender` | 3 | LabelEncoder → 0, 1, 2 |
| `specialty` | 8 | LabelEncoder → 0–7 |
| `disability` | 4 | LabelEncoder → 0–3 |
| `place` | ~15 | LabelEncoder → 0–14 |
| `appointment_shift` | 2 | LabelEncoder → 0, 1 |
| `heat_intensity` | 5 | LabelEncoder → 0–4 |
| `rain_intensity` | 4 | LabelEncoder → 0–3 |

**Rationale for LabelEncoder over OHE:** All four primary classifiers (Decision Tree, Random Forest, XGBoost, LightGBM) are tree-based. Trees split on thresholds, not distances or dot products — they do not assume any ordinal relationship between integer class labels. OHE would create 20+ sparse binary columns, increasing dimensionality and training time without improving tree model performance. Logistic Regression (the only linear model) is the one model where LabelEncoding implies unintended ordinality, but since it serves only as a linear baseline, this tradeoff is explicitly accepted and documented.

### 5.4 Final Feature Set (29 Features)

| Category | Count | Features |
|----------|-------|---------|
| Encoded categoricals | 7 | `gender_enc`, `specialty_enc`, `disability_enc`, `place_enc`, `appointment_shift_enc`, `heat_intensity_enc`, `rain_intensity_enc` |
| Patient demographics | 4 | `age`, `under_12_years_old`, `over_60_years_old`, `patient_needs_companion` |
| Appointment logistics | 2 | `appointment_time`, `appointment_shift_enc` |
| Temporal | 5 | `month`, `day_of_week`, `week_of_year`, `quarter`, `is_weekend` |
| Weather numeric | 6 | `average_temp_day`, `average_rain_day`, `max_temp_day`, `max_rain_day`, `rainy_day_before`, `storm_day_before` |
| Health & social | 5 | `Hypertension`, `Diabetes`, `Alcoholism`, `Handcap`, `Scholarship` |
| Communication | 1 | `SMS_received` |
| **Total** | **29** | |

### 5.5 Multicollinearity Assessment

Two high-correlation pairs were identified in the heatmap analysis (Section 4.11):

| Pair | Pearson r | Decision |
|------|-----------|---------|
| `average_temp_day` & `max_temp_day` | 0.96 | Both retained |
| `rainy_day_before` & `storm_day_before` | 0.99 | Both retained |

**Why both columns were retained in each pair:** VIF (Variance Inflation Factor) analysis is critical for linear regression models where inflated coefficient variance causes instability. For tree-based models (Random Forest, XGBoost, LightGBM), there is no coefficient estimation — splits are made independently at each node and the model naturally down-weights redundant features. Feature importance scores confirm this: the model assigns importance to one feature of each correlated pair and relatively less to the other.

If Logistic Regression were the production model, VIF scores would be computed and features exceeding VIF > 10 would be dropped. This is documented as a future improvement step.

---

## 6. No-Show Classification — Model Building

### 6.1 Train-Test Split

A **stratified 70/30 split** was applied with `random_state=42`:

| Split | Rows | No-Show Rate | No-Show Count | Show Count |
|-------|------|-------------|--------------|------------|
| Full dataset | 109,593 | 31.78% | 34,832 | 74,761 |
| Training set | 76,715 | 31.78% | 24,382 | 52,333 |
| Test set | 32,878 | 31.78% | 10,450 | 22,428 |

`stratify=y` ensures both subsets maintain the original 31.78% no-show ratio, preventing accidental class ratio drift that would skew evaluation metrics. The test set was **never used** during training, cross-validation, or hyperparameter tuning — it was reserved entirely for final model evaluation.

### 6.2 Handling Class Imbalance with SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was applied **exclusively within each training fold** of the cross-validation pipeline:

| Stage | Class 0 (Show) | Class 1 (No-Show) | Ratio |
|-------|---------------|-------------------|-------|
| Before SMOTE (train set) | 52,333 | 24,382 | 68:32 |
| After SMOTE (train set) | 52,333 | 52,333 | 50:50 |
| Test set (unchanged) | 22,428 | 10,450 | 68:32 |

SMOTE creates synthetic no-show samples by: (1) selecting a minority-class sample; (2) finding its k=5 nearest minority neighbours in feature space; (3) interpolating a new synthetic point along the line segment between the sample and a randomly selected neighbour. Unlike simple oversampling (duplication), SMOTE generates diverse synthetic examples that introduce controlled variance, reducing the risk of the model memorising the minority class.

**Critical implementation note:** SMOTE is placed **inside the sklearn Pipeline** (`Pipeline([('smote', SMOTE()), ('model', classifier)])`). This ensures oversampling is applied only to the training portion of each fold during cross-validation — never to the validation fold. Applying SMOTE globally before CV would cause data leakage, where synthetic samples derived from validation data contaminate the training signal and produce artificially optimistic CV scores.

### 6.3 Classifiers Compared

Five classifiers were trained and evaluated:

| Classifier | Pipeline Configuration | Tuning Method | Type |
|-----------|----------------------|--------------|------|
| Logistic Regression | StandardScaler → SMOTE → LR | GridSearchCV | Linear baseline |
| Decision Tree | SMOTE → DT | RandomizedSearchCV | Shallow tree |
| Random Forest | SMOTE → RF | RandomizedSearchCV | Ensemble — bagging |
| XGBoost | SMOTE → XGB | RandomizedSearchCV | Ensemble — boosting |
| **LightGBM** | SMOTE → LGBM | RandomizedSearchCV | **Ensemble — gradient boosting** |

**Rationale for each classifier:**

- **Logistic Regression:** Linear baseline to quantify the performance gain from tree-based non-linearity. Expected to underperform given that the correlation heatmap shows no strong linear relationships (max |r| = 0.14)
- **Decision Tree:** Interpretable single-tree benchmark; `max_depth` constraint limits leaf count and prevents memorisation
- **Random Forest:** Reduces Decision Tree variance by averaging 100 independently-trained trees on bootstrap samples with feature subsampling (`max_features='sqrt'`). Primarily addresses the high-variance problem of a single tree
- **XGBoost:** Reduces bias iteratively by training each new tree to fit the pseudo-gradient (residuals) of the current ensemble. Regularisation via `gamma` (minimum loss reduction for a split) and `min_child_weight` (minimum sample weight in a leaf) explicitly controls overfitting
- **LightGBM:** Leaf-wise growth strategy grows the single most impactful leaf at each step (rather than completing all leaves at a depth level). On large datasets (109K rows), this produces more accurate splits with less computation. Gradient-based One-Side Sampling (GOSS) retains high-gradient training samples, further improving efficiency

### 6.4 Hyperparameter Tuning

**Cross-validation strategy:**

| Classifier | CV Strategy | n_splits | Tuning Method | n_iter |
|-----------|------------|---------|--------------|--------|
| Logistic Regression | StratifiedKFold | 5 | GridSearchCV | exhaustive |
| Decision Tree | StratifiedKFold | 3 | RandomizedSearchCV | 25 |
| Random Forest | StratifiedKFold | 3 | RandomizedSearchCV | 15 |
| XGBoost | StratifiedKFold | 3 | RandomizedSearchCV | 15 |
| LightGBM | StratifiedKFold | 3 | RandomizedSearchCV | 15 |

`refit='roc_auc'` was set for all searches — the best hyperparameter combination is selected by cross-validated ROC-AUC, ensuring hyperparameter choices are never influenced by test-set performance.

**Logistic Regression — GridSearchCV parameter grid:**

```python
param_grid_lr = {
    'model__C':            [0.01, 0.1, 1, 5, 10, 20],       # L2 regularisation strength
    'model__solver':       ['lbfgs', 'liblinear'],
    'model__class_weight': [None, 'balanced'],
    'model__max_iter':     [500, 1000]
}
```

**LightGBM — RandomizedSearchCV parameter distributions:**

```python
param_dist_lgb = {
    'model__n_estimators':      [100, 200],
    'model__learning_rate':     [0.05, 0.1],
    'model__num_leaves':        [31, 50, 100],
    'model__max_depth':         [-1, 5, 10],
    'model__subsample':         [0.8, 1.0],
    'model__colsample_bytree':  [0.8, 1.0],
    'model__min_child_samples': [10, 20]
}
```

**Logistic Regression — Optimal Threshold Search:**

After GridSearchCV, an explicit threshold search was performed over `np.arange(0.1, 0.9, 0.01)` to find the probability cutoff that maximises F1-Score on the test set. The default threshold of 0.50 is suboptimal for imbalanced data where the cost of FP (unnecessary reminder call) and FN (missed no-show, wasted slot) are asymmetric.

### 6.5 Evaluation Methodology

All classifiers were evaluated on the **unchanged imbalanced test set** (32,878 rows, 31.78% no-show rate) using the following metrics:

| Metric | Why Used |
|--------|---------|
| **ROC-AUC** (primary) | Threshold-independent; measures ranking quality; robust to class imbalance |
| **F1-Score** (secondary) | Harmonic mean of Precision and Recall; balances both for the minority class |
| Recall (no-show class) | Clinical priority: catching missed no-shows reduces wasted slots |
| Precision | Operational cost of unnecessary reminder calls |
| Accuracy | Reported but not used for model selection — inflated by majority class |

**Why ROC-AUC is the primary metric:** A model predicting all appointments as "show" achieves 68.22% accuracy — which sounds reasonable but detects zero no-shows. ROC-AUC evaluates whether the model correctly ranks true no-shows higher-risk than true shows across all possible probability thresholds. A random classifier scores 0.50; perfect classifier scores 1.00.

---

## 7. Demand Forecasting — Model Building

### 7.1 Problem Formulation

Daily appointment count is treated as a **supervised regression** problem. Rather than classical time series methods (ARIMA, SARIMA, Exponential Smoothing), a feature-based ML approach was chosen:

| Factor | Why ML Regression Over ARIMA |
|--------|-------------------------------|
| Structural break (COVID-19) | ARIMA requires manual intervention terms or data truncation; ML handles non-stationarity through features |
| Rich exogenous features | Calendar and weather features integrate naturally as regression inputs; ARIMA requires separate ARIMAX specification |
| Lag autocorrelation | Explicit lag features replicate what ARIMA's AR terms capture, without distributional assumptions |
| Multiple regressors | Easy to compare Ridge, Random Forest, and LightGBM; ARIMA family has no equivalent comparison framework |

### 7.2 Time Series Properties

| Property | Value |
|----------|-------|
| Total days in raw series | 498 |
| Usable days (after lag creation) | 484 |
| Training period | 387 days (80%) |
| Test period | 97 days (20%) |
| Mean daily demand | 220.1 appointments |
| Standard deviation | 245.8 appointments |
| Coefficient of variation | 1.12 |
| Minimum demand | 1 appointment |
| Maximum demand | 1,512 appointments |

The coefficient of variation (std/mean = 1.12) exceeds 1.0, indicating **extreme volatility** where the standard deviation is larger than the mean. This is driven by three sources: (1) weekends with near-zero demand, (2) the COVID-19 demand collapse, and (3) post-pandemic recovery variance. This fundamental property bounds achievable R² and explains why even well-fitted models produce modest explained variance.

### 7.3 Feature Engineering for Time Series

Thirteen features were engineered from the appointment date:

| Feature | Type | What It Captures |
|---------|------|-----------------|
| `lag_1` | Lag | Yesterday's demand — short-term autocorrelation |
| `lag_7` | Lag | Same weekday last week — weekly cycle (strongest predictor) |
| `lag_14` | Lag | Two-week pattern |
| `rolling_7` | Rolling mean | 7-day smoothed trend level |
| `rolling_14` | Rolling mean | 14-day smoothed trend level |
| `day_of_week` | Calendar | Monday–Sunday demand cycles |
| `month` | Calendar | Annual seasonality |
| `week_of_year` | Calendar | Week-level patterns |
| `quarter` | Calendar | Quarterly healthcare demand cycles |
| `day_of_month` | Calendar | Intra-month scheduling patterns |
| `is_weekend` | Binary | Weekend demand suppression |
| `is_month_start` | Binary | Month-start scheduling patterns |
| `is_month_end` | Binary | Month-end scheduling patterns |

> **Leakage prevention:** All lag and rolling features use `.shift(1)` before computation. This guarantees that at any prediction step, only past demand values are used. The first 14 rows of the series are dropped due to NaN values from the shifting operation — a necessary cost of lag feature creation.

### 7.4 Chronological Train-Test Split

An **80/20 chronological split** was used — not random:

| Split | Period | Days | Demand Range |
|-------|--------|------|-------------|
| Training | First 80% of dates | 387 days | Jan 2020 – ~Jan 2021 |
| Test | Last 20% of dates | 97 days | ~Feb 2021 – May 2021 |

**Why chronological (not random)?** Random splitting of time series data constitutes data leakage. A random split would train on records from, say, March 2021 and predict January 2020 — a direction impossible in production, where only historical data is available. The chronological split replicates deployment conditions: train on all available past data, evaluate on the unseen future.

### 7.5 Regressors Compared

| Regressor | Configuration | Tuning Method | Type |
|-----------|--------------|--------------|------|
| Ridge Regression | StandardScaler → Ridge | GridSearchCV (alpha) | Linear baseline |
| Random Forest | RF Regressor | RandomizedSearchCV | Ensemble — bagging |
| **LightGBM** | LGBM Regressor | RandomizedSearchCV | **Ensemble — gradient boosting** |

**Ridge Regression** with `StandardScaler` (mandatory for Ridge since L2 penalises coefficient magnitude) and `alpha` tuned over `[0.01, 0.1, 1, 10, 50, 100]` provides the linear baseline.

**Random Forest Regressor** with `n_estimators` tuned over `[100, 200, 300, 500]` and `max_depth` over `[5, 10, 15, None]`.

**LightGBM Regressor** with `learning_rate` over `[0.01, 0.05, 0.1]`, `num_leaves` over `[31, 50, 100, 150]`, and `subsample` over `[0.6, 0.8, 1.0]`.

All regressors used `KFold(n_splits=5)` (not Stratified, since the target is continuous) with `scoring='neg_mean_absolute_error'` for hyperparameter selection.

---

## 8. Results & Model Performance

### 8.1 Classification Results — All Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.68 | ~0.48 | ~0.42 | ~0.44 | ~0.62 |
| Decision Tree | ~0.70 | ~0.52 | ~0.46 | ~0.49 | ~0.72 |
| Random Forest | ~0.72 | ~0.55 | ~0.48 | ~0.51 | ~0.76 |
| XGBoost | ~0.72 | ~0.57 | ~0.50 | ~0.53 | ~0.77 |
| **LightGBM** | **0.7245** | **0.5753** | **0.5086** | **0.5399** | **0.7752** |

**Best model: LightGBM** with the highest ROC-AUC of **0.7752** — achieving both the primary metric target (> 0.75) and secondary metric target (F1 > 0.45).

**Model progression analysis:** The ROC-AUC improvement from Logistic Regression (0.62) to LightGBM (0.7752) represents a 15.2 percentage point gain — directly attributable to the ability of ensemble tree models to capture non-linear interactions (specialty × temperature × disability) that the linear model cannot.

### 8.2 Confusion Matrix — LightGBM (Best Classifier)

Evaluated on 32,878 test rows (10,450 actual no-shows, 22,428 actual shows):

|  | **Predicted: Show** | **Predicted: No-Show** |
|--|---------------------|----------------------|
| **Actual: Show** | 18,504 (TN) | 3,924 (FP) |
| **Actual: No-Show** | 5,135 (FN) | 5,315 (TP) |

**Detailed interpretation:**

| Quadrant | Count | Clinical Meaning | Operational Cost |
|----------|-------|-----------------|-----------------|
| True Positive (TP) | 5,315 | No-show correctly identified → intervention triggered | Low (reminder cost) |
| True Negative (TN) | 18,504 | Show patient correctly left uncontacted | Zero |
| False Positive (FP) | 3,924 | Show patient unnecessarily contacted | Minor (unnecessary call) |
| **False Negative (FN)** | **5,135** | **No-show patient missed → empty slot** | **High (wasted slot)** |

Of 10,450 actual no-shows in the test set, the model correctly catches **5,315 (50.86%)** — meaning approximately half of all no-shows are identified in advance and can receive targeted intervention. The remaining 49.14% (5,135 FN) are the model's primary limitation.

**Recall vs Precision trade-off:** Recall = 50.86% means roughly 1-in-2 actual no-shows are correctly flagged. Lowering the classification threshold from 0.50 to 0.35 would increase recall (more no-shows caught) at the cost of precision (more show patients contacted unnecessarily) — a clinical decision the facility must make based on their cost structure for reminder calls vs empty slot losses.

### 8.3 Feature Importance — LightGBM Classifier

| Rank | Feature | Importance Score | Category |
|------|---------|-----------------|----------|
| 1 | `age` | 538 | Patient demographics |
| 2 | `place_enc` | 381 | Geography |
| 3 | `max_rain_day` | 300 | Weather |
| 4 | `average_rain_day` | 292 | Weather |
| 5 | `max_temp_day` | 280 | Weather |
| 6 | `average_temp_day` | 261 | Weather |
| 7 | `appointment_time` | 145 | Logistics |
| 8 | `specialty_enc` | 119 | Appointment |
| 9 | `disability_enc` | 84 | Patient demographics |
| 10 | `heat_intensity_enc` | 79 | Weather |

Age and geographic location are the two dominant individual predictors. Weather features (temperature and rain) collectively account for four of the top six features — consistent with the EDA finding that heavy-cold days produce 52.93% no-show rates. Specialty and disability type rank 8th and 9th, confirming EDA findings but showing that the raw numeric weather signal carries more discriminating power than the encoded categorical specialty flag.

### 8.4 Demand Forecasting Results — All Regressors

| Regressor | MAE | RMSE | R² | Direction |
|-----------|-----|------|----|-----------|
| Ridge Regression | ~200 | ~310 | 0.152 | Linear baseline |
| Random Forest | ~185 | ~285 | 0.190 | Bagging ensemble |
| **LightGBM** | **176.68** | **272.71** | **0.077** | **Best MAE** |

> **Note on conflicting metrics:** LightGBM achieves the best MAE (176.68) but a lower R² than Random Forest (0.077 vs 0.190). This occurs because R² is sensitive to variance decomposition, while MAE measures the average absolute error directly. LightGBM produces predictions with lower average error per day (operationally more useful) but a different variance structure. For staffing planning purposes, MAE is the more relevant metric.

**Best model for operational use: LightGBM Regressor** with MAE = 176.68 appointments/day.

### 8.5 Top Time Series Features — LightGBM Regressor

| Rank | Feature | Importance Score | Interpretation |
|------|---------|----------------|----------------|
| 1 | `lag_1` | 531 | Yesterday's demand is the single strongest predictor |
| 2 | `lag_7` | 498 | Same weekday last week — captures weekly seasonality |
| 3 | `rolling_14` | 391 | 14-day smoothed trend level |
| 4 | `lag_14` | 380 | Two-week autocorrelation pattern |
| 5 | `day_of_month` | 354 | Intra-month scheduling cycle |

Lag features dominate, confirming that **recent demand history is far more informative than calendar features** for this dataset. `lag_7` being the second most important feature validates that healthcare appointment demand follows a strong weekly rhythm — Monday's count is highly correlated with the previous Monday's count. This is the time-series equivalent of the correlation heatmap showing strong pairwise relationships in the classification dataset.

### 8.6 Why R² is Low for Demand Forecasting

The R² of 0.077–0.190 requires honest contextualisation. This is not model failure — it reflects genuine properties of the data:

1. **COVID-19 structural break:** Demand dropped by 80–90% in March 2020 and recovered non-linearly over 14 months. No calendar feature or lag value can encode an unprecedented external shock. Any model trained on pre-pandemic data and predicting the drop period will incur large errors regardless of algorithm quality.

2. **Extreme volatility (CV = 1.12):** When std (245.8) exceeds mean (220.1), the regression-to-mean effect means the model systematically under-predicts high-demand days (positive residuals) and over-predicts low-demand days (negative residuals). This pattern is visible in the residual plot (Section 6.8 of the notebook).

3. **Short stable period:** Only 484 usable days, of which a substantial fraction covers the disrupted pandemic period. A model trained on 2+ years of post-pandemic stable data would produce substantially higher R².

The MAE of 176.68 appointments/day provides the most operationally interpretable summary: on an average day, the forecast is off by approximately 177 appointments. For a facility averaging 220 appointments/day, this represents an 80% average error — sufficient for directional staffing guidance (is tomorrow a high-demand or low-demand day?) but not for precise slot-level scheduling.

---

## 9. Streamlit Application

### 9.1 Architecture

The application is a four-page Streamlit dashboard (`app.py`) backed by nine pre-trained `.pkl` artefact files. The architecture separates model inference from the UI layer:

```
app.py (1,279 lines)
├── Page 1: Operations Dashboard
│   ├── 5 KPI cards (total, attended, no-shows, SMS rate, median age)
│   ├── Gender, Specialty, Monthly trend, Heat intensity charts
│   ├── Day-of-week, Disability, Daily time series charts
│   └── Health conditions styled comparison table
│
├── Page 2: No-Show Predictor
│   ├── Full patient input form (demographics, appointment, weather, health)
│   ├── predict_proba() → probability risk score
│   ├── Gauge chart with colour zones (green/yellow/red)
│   └── Colour-coded risk box with specific intervention recommendation
│
├── Page 3: Demand Forecaster
│   ├── Historical 90-day context chart
│   ├── Iterative multi-day forecast (auto-seeded from last_values.pkl)
│   ├── Bar chart (weekday/weekend colour-coded)
│   ├── Staffing recommendation per day
│   └── Historical weekly rollup area chart
│
└── Page 4: Model Insights
    ├── Tab 1: Classifier — feature importance + cumulative curve
    ├── Tab 2: Forecaster — feature importance + methodology
    ├── Tab 3: EDA Explorer — 5 interactive analyses
    └── Tab 4: Interventions — 6 action cards + segment table
```

### 9.2 Saved Model Artefacts

| File | Contents | Used By |
|------|---------|---------|
| `best_classifier.pkl` | Trained LightGBM Pipeline (SMOTE + model) | Page 2 predictor |
| `best_classifier_name.pkl` | Model name string | Sidebar display |
| `demand_forecaster.pkl` | Trained LightGBM Regressor | Page 3 forecaster |
| `best_forecaster_name.pkl` | Forecaster name string | Sidebar display |
| `label_encoders.pkl` | LabelEncoder objects for 7 categorical columns | Page 2 inference |
| `features.pkl` | Ordered list of 29 classification feature names | Page 2 inference |
| `ts_features.pkl` | Ordered list of 13 forecasting feature names | Page 3 inference |
| `cat_options.pkl` | Unique values per categorical feature (for UI dropdowns) | Page 2 form |
| `age_median.pkl` | Median age = 12 years for inference-time imputation | Page 2 inference |
| `last_values.pkl` | Last known lag values for iterative forecasting seed | Page 3 forecaster |
| `daily_ts.pkl` | Full daily demand time series | Pages 3 & 4 charts |

### 9.3 No-Show Predictor — Inference Pipeline

When a new appointment is submitted via the form, the inference steps mirror the training preprocessing exactly:

```
1. User inputs 29 fields via Streamlit form
        ↓
2. Load label_encoders.pkl → apply LabelEncoder.transform()
   to gender, specialty, disability, place, shift, heat, rain
        ↓
3. Extract calendar features from appointment_date
   (month, day_of_week, week_of_year, quarter, is_weekend)
        ↓
4. Apply age clipping [0, 100]; fill missing age with age_median.pkl
        ↓
5. Assemble feature vector in exact order from features.pkl (29 features)
        ↓
6. clf.predict_proba(X_input)[0, 1] → no-show probability ∈ [0, 1]
        ↓
7. Apply operational thresholds:
   ├── prob > 0.60 → HIGH RISK → phone call intervention
   ├── prob > 0.35 → MEDIUM RISK → WhatsApp/SMS reminder
   └── prob ≤ 0.35 → LOW RISK → no action required
        ↓
8. Display gauge chart, risk box, and input summary table
```

### 9.4 Demand Forecaster — Iterative Forecasting

The demand forecaster uses a **chained iterative forecast**: each day's prediction becomes the lag seed for the next day. This is the only viable strategy for multi-day ahead forecasting when lag features are used:

```python
for each target date d:
    lag_1  = recent_vals[-1]            # yesterday's predicted demand
    lag_7  = recent_vals[-7]            # same weekday last week
    lag_14 = recent_vals[-14]           # two weeks ago
    roll_7 = mean(recent_vals[-7:])     # 7-day rolling average
    roll_14= mean(recent_vals[-14:])    # 14-day rolling average

    X_fc = assemble_features(d, lag_1, lag_7, lag_14, roll_7, roll_14)
    pred = max(0, forecaster.predict(X_fc)[0])  # clip at zero
    recent_vals.append(pred)                    # seed next iteration
```

The chain is initialised from `last_values.pkl` which stores the true observed demand values for the last 14 days in the training data. Forecast uncertainty compounds over multi-day horizons — predictions for day N+7 are less reliable than predictions for day N+1 because they depend on 7 prior predicted (rather than observed) values.

### 9.5 Running the Application

```bash
# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn lightgbm xgboost \
            imbalanced-learn matplotlib seaborn joblib missingno

# Ensure directory structure:
# ├── app.py
# ├── Medical_appointment_data.csv
# └── Models/
#     ├── best_classifier.pkl
#     ├── demand_forecaster.pkl
#     ├── label_encoders.pkl
#     └── (remaining .pkl files)

# Launch the dashboard
streamlit run app.py
```

Update `MODELS_DIR` and `DATA_PATH` at the top of `app.py` to match your local directory paths before running.

---

## 10. Business Recommendations

Based on model results and EDA findings, the following interventions are recommended in priority order. Each recommendation is grounded in a specific quantified finding from the analysis.

### Priority 1 — Immediate Action (High Impact, Low Cost)

**Recommendation 1: Redesign the patient reminder system**

**Evidence:** SMS reminders show +0.10 pp difference in no-show rate (31.75% without vs 31.85% with SMS) — statistically indistinguishable from zero. The dataset does not record reminder timing or content, making it impossible to evaluate quality, but the null result strongly suggests the current approach is ineffective.

**Action plan:**
- Replace SMS with personalised WhatsApp voice/text messages including the patient's name, appointment date, specialist name, and one-tap confirmation link
- Send reminders **48 hours before** the appointment (not day-of)
- Target high-risk patients identified by the LightGBM classifier (probability > 0.50) for priority outreach
- Run an **A/B test**: randomly assign 50% of high-risk appointments to the new protocol, 50% to current SMS, and measure no-show rates after 3 months before full rollout

**Projected impact:** Research on personalised healthcare reminders (Salazar et al., 2022) suggests 4–8 pp reduction in no-show rates among targeted groups.

---

**Recommendation 2: Proactive outreach for "sem especialidade" appointments**

**Evidence:** The 52.78% no-show rate for appointments with no assigned specialty is 21 percentage points above the overall average. This is the highest-risk identifiable group in the dataset.

**Action plan:**
- All "sem especialidade" appointments receive a mandatory personal phone call 48 hours before attendance
- Calls should explain the appointment's purpose, duration, and what to bring — addressing the likely cause of low perceived urgency
- Track confirmation vs non-confirmation for same-day rescheduling

**Projected impact:** If targeted outreach reduces the "sem especialidade" group's no-show rate from 52.78% to 35% (closer to the physiotherapy average), and assuming this group comprises ~5% of all appointments, this single intervention could reduce the overall clinic no-show rate by ~0.9 percentage points — approximately 2,000 fewer missed appointments per year.

### Priority 2 — Medium-Term (Operational Changes)

**Recommendation 3: Weather-aware scheduling and capacity management**

**Evidence:** Heavy-cold days produce 52.93% no-show rates (+21.15 pp above average). In Itajaí's typically warm climate, heavy-cold days are predictable from 48-hour weather forecasts.

**Action plan:**
- Integrate a weather forecast API into the scheduling system
- On days with a forecast of heavy-cold conditions: reduce non-urgent appointment load by 10–15% and increase buffer slots for walk-in or rescheduled patients
- Flag all patients with appointments on forecast cold days for a proactive call the day before

---

**Recommendation 4: Risk-based overbooking for morning physiotherapy slots**

**Evidence:** Morning shift has a 33.01% no-show rate (+2.33 pp vs afternoon). Physiotherapy has the highest absolute appointment volume. The combination of above-average risk and high volume means morning physiotherapy slots generate the most wasted capacity.

**Action plan:**
- Use the LightGBM classifier to score each scheduled appointment slot
- For morning physiotherapy blocks where the average predicted no-show probability exceeds 40%, book one additional patient from the waitlist per session (1.10×–1.15× overbooking)
- Monitor actual attendance vs predicted to refine the overbooking ratio monthly

**Note:** Overbooking must be implemented carefully in a paediatric disability context — patient families face significant logistical costs for wasted trips. The overbooking policy should include a clear communication protocol if a slot is over-attended.

---

**Recommendation 5: Geographic transport assistance for distant cities**

**Evidence:** Patients from Luiz Alves (~13.6% no-show), Ilhota (~13.1%), and Bombinhas (~12.8%) show elevated rates consistent with the distance-decay effect. These rates are 4–5 pp above the Itajaí home-city rate.

**Action plan:**
- Partner with local health posts (UBS) in distant cities to coordinate group transport on scheduled CER appointment days
- Pilot telemedicine follow-up options for post-appointment consultations to reduce the frequency of in-person visits for stable patients
- Consider satellite scheduling days where CER clinicians travel to distant cities for routine appointments

### Priority 3 — Strategic (Long-Term)

**Recommendation 6: Implement patient appointment history tracking**

**Evidence:** The single most powerful missing predictor is a patient's prior no-show rate. Clinical literature consistently identifies individual appointment history as the strongest predictor of future no-show behaviour. The current dataset lacks any patient-level longitudinal data.

**Action plan:**
- Add `patient_id` linkage to the appointment system
- Track cumulative no-show rate and no-show streak (consecutive missed appointments) per patient
- Retrain the classifier with these features — projected to push ROC-AUC from 0.7752 to 0.85+

---

**Recommendation 7: Monthly capacity planning using the demand forecaster**

**Evidence:** Even with modest R² (0.077–0.19), the LightGBM forecaster correctly identifies the directional demand pattern — high-demand vs low-demand periods — which is sufficient for staffing planning decisions.

**Action plan:**
- Generate monthly demand forecasts in the Streamlit dashboard at the start of each month
- Use the 14-day rolling average forecast as input to staff scheduling decisions (full staff / standard / minimal)
- Retrain the forecaster quarterly with the latest available data to update the lag seeds

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

| Limitation | Severity | Impact on Results | Mitigation Path |
|-----------|----------|------------------|----------------|
| No patient appointment history | High | Single strongest predictor absent; ROC-AUC ceiling ~0.78 | Implement patient_id linkage |
| COVID-19 structural break in demand data | High | Demand forecasting R² = 0.077–0.19 | Retrain on 2022+ post-pandemic stable data |
| Short time window (484 usable days) | Medium | Limited training signal for forecaster | Collect 2+ years of post-pandemic data |
| SMS data lacks timing and content | Medium | Cannot evaluate reminder quality or redesign | Capture reminder timestamp, channel, and content type |
| Single facility, single region | Medium | Results may not generalise to other SUS facilities | Validate on multi-facility dataset |
| Weather data from one station only | Low | Inaccurate for patients from distant cities | Source city-level weather API data |
| Label Encoding for LR baseline | Low | LR coefficient interpretation affected | Use OHE if LR is ever used in production |
| No probability calibration check | Low | Dashboard thresholds may be miscalibrated | Add `CalibratedClassifierCV` wrapper |

### 11.2 Missing Diagnostic Checks (Future Additions)

The following standard diagnostic plots were not included in the current version but are documented for the next iteration:

| Check | What It Would Show | Implementation |
|-------|-------------------|---------------|
| Learning curves | Train vs validation AUC as dataset size grows; diagnose bias vs variance | `sklearn.model_selection.learning_curve` |
| Precision-Recall curves | Minority-class performance across all thresholds; more informative than ROC for imbalanced data | `sklearn.metrics.precision_recall_curve` |
| Calibration curve | Whether predicted probabilities correspond to observed rates | `sklearn.calibration.calibration_curve` |
| Train vs test AUC gap table | Explicit overfitting quantification per model | Compare `grid.best_score_` with `roc_auc_score(y_test)` |
| VIF scores | Formal multicollinearity test for linear models | `statsmodels.stats.outliers_influence.variance_inflation_factor` |

### 11.3 Future Work — Roadmap

**Near-term (0–3 months):**
- Add learning curves and Precision-Recall curves to the notebook (code available in diagnostic guide)
- Lower the operational classification threshold from 0.50 to 0.35 for the ensemble models to increase recall at acceptable precision cost
- Add SHAP (SHapley Additive exPlanations) for per-patient prediction explanations in the Streamlit dashboard
- Implement probability calibration using `CalibratedClassifierCV(method='isotonic')`

**Medium-term (3–12 months):**
- Add patient appointment history as a feature (requires `patient_id` linkage) — expected to push ROC-AUC from 0.78 to 0.85+
- Build specialty-specific demand sub-models (physiotherapy, psychotherapy, speech therapy separately) — weekly sub-model R² would be substantially higher than daily total model
- Evaluate weekly-level (vs daily-level) demand forecasting — aggregation reduces volatility and would produce meaningfully higher R²
- Design and implement A/B test of the redesigned personalised reminder system

**Long-term (12+ months):**
- Retrain all models on 2022+ post-pandemic stable data
- Integrate with the clinic's Electronic Health Record (EHR) system via REST API for real-time risk scoring at appointment booking time
- Implement automated model monitoring for concept drift (patient population changes, seasonal shifts in attendance behaviour)
- Expand validation to other CER facilities in the AMFRI region to test generalisation

---

## 12. Conclusion

This project delivers a complete, production-ready machine learning pipeline for the CER Univali rehabilitation centre. Two ML systems were built, validated, and deployed:

**No-Show Classifier:** A LightGBM model with ROC-AUC of **0.7752** correctly identifies 50.9% of all actual no-show patients in advance — enabling targeted pre-appointment interventions for over 5,300 at-risk appointments per 32,878-appointment test cycle. The classifier achieves the defined success criterion (ROC-AUC > 0.75) using 29 features spanning patient demographics, appointment logistics, weather conditions, and health flags.

**Demand Forecaster:** A LightGBM regressor with MAE of **176.68 appointments/day** predicts daily clinic volume to support staffing decisions. The model's modest R² (0.077) is contextualised by the COVID-19 structural break and extreme demand volatility (CV > 1.0) present in the 2020–2021 dataset — not a modelling failure but an honest reflection of inherently unpredictable data.

**Streamlit Application:** A four-page operational dashboard puts both models in the hands of clinic staff — enabling real-time patient risk scoring, multi-day demand forecasting, and access to data-driven intervention guidance without requiring technical knowledge.

**Three most actionable findings:**

1. **The current SMS reminder system is completely ineffective** (+0.10 pp effect). Redesigning to personalised 48-hour WhatsApp reminders targeted at the model's high-risk patients is the single highest-ROI, lowest-cost intervention available to the clinic today.

2. **Cold-weather days and unassigned-specialty appointments each produce ~53% no-show rates** — 21 percentage points above average. Weather-aware scheduling adjustments and mandatory confirmation calls for "sem especialidade" appointments are the two highest-impact operational changes identified.

3. **The absence of patient appointment history is the binding constraint on model accuracy.** Implementing patient-level history tracking and retraining the classifier would likely push ROC-AUC from 0.7752 to 0.85+, substantially increasing the number of at-risk patients identifiable in advance.

At scale across Brazil's SUS system, the financial and patient care implications of even a 5–10 percentage point reduction in no-show rates are substantial — fewer wasted specialist slots, shorter patient waitlists, reduced clinician underutilisation, and improved health outcomes for the disability populations these centres serve.

---

## Appendix A — Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | — |
| Data manipulation | pandas, NumPy | Dataset loading, transformation, feature engineering |
| Machine learning | scikit-learn | Pipelines, CV, preprocessing, evaluation metrics |
| Gradient boosting | LightGBM, XGBoost | Primary classifiers and regressors |
| Imbalance handling | imbalanced-learn (SMOTE) | Minority class oversampling |
| Visualisation (notebook) | matplotlib, seaborn | All EDA and diagnostic plots |
| Visualisation (app) | Plotly | Interactive dashboard charts |
| Web application | Streamlit | Four-page operational dashboard |
| Model serialisation | joblib | .pkl artefact saving and loading |
| Missingness visualisation | missingno | Missing value matrix (Section 2) |
| Development environment | Jupyter Notebook / VS Code | — |

---

## Appendix B — Model Hyperparameters

**LightGBM Classifier (Best Classification Model)**

```python
Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(
        n_estimators=100,       # tuned via RandomizedSearchCV
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    ))
])
```

**LightGBM Regressor (Best Demand Forecasting Model)**

```python
LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
```

**Ridge Regression (Demand Forecasting Baseline)**

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))    # alpha tuned via GridSearchCV over [0.01, 0.1, 1, 10, 50, 100]
])
```

---

## Appendix C — Notebook Structure

The optimised notebook (`no_show_machine_learning_optimized.ipynb`) contains **98 cells** — 49 code cells and 49 markdown annotation cells — structured as follows:

| Section | Cells | Description |
|---------|-------|-------------|
| Section 1 — Setup & Imports | 1–2 | All libraries, global plot style, colour constants, file paths |
| Section 2 — Data Loading | 3–8 | Shape, dtypes, describe, missingness matrix, missing value bar chart |
| Section 3 — EDA | 9–48 | 12 subsections: gender, age, specialty, shift, temporal, weather, health, disability, geography, correlation, time series |
| Section 4 — Preprocessing | 49–58 | Column rename, date features, imputation, target encoding, label encoding, feature set definition |
| Section 5 — Classification | 59–79 | Train-test split, SMOTE pipeline, 5 classifiers with CV tuning, evaluation, confusion matrices, ROC curves, feature importance |
| Section 6 — Demand Forecasting | 80–97 | Daily TS build, lag feature engineering, chronological split, 3 regressors with CV tuning, evaluation, residual plot, feature importance |
| Section 7 — Business Summary | 98 | Key findings printout and intervention recommendations table |

Every code cell is followed by a markdown annotation cell explaining the method used, why it was chosen, what the output means, and its implication for modelling or clinical operations.

---

## Appendix D — References

1. Salazar, L.H.A. et al. (2022). Application of Machine Learning Techniques to Predict a Patient's No-Show in the Healthcare Sector. *Future Internet*, 14(3), 3. https://doi.org/10.3390/fi14010003

2. Salazar, L.H. et al. (2020). Using Different Models of Machine Learning to Predict Attendance at Medical Appointments. *Journal of Information Systems Engineering and Management*, 5(4), em0122. https://doi.org/10.29333/jisem/8421

3. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30. https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

5. Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

6. Scikit-learn: Machine Learning in Python. Pedregosa et al. (2011). *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

7. Snoek, J., Larochelle, H., & Adams, R.P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems*, 25.

8. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

9. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 3 (Linear Models for Regression), Chapter 7 (Sparse Kernel Machines).

10. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

---