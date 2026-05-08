# Medical Appointment No-Show Prediction & Demand Forecasting

## Project Overview
This project focuses on predicting patient attendance and optimizing scheduling using data-driven insights to improve operational efficiency and care delivery.

---

## Deliverables
| File | Description |
|------|-------------|
| `medical_noshow_notebook.ipynb` | Full EDA → Preprocessing → Classification → Forecasting notebook |
| `app.py` | Streamlit multi-page web application |
| `Models/` | Saved model files (.pkl) |

---

## Quick Start

### Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn joblib
```

### Run the Streamlit App
```bash
streamlit run app.py
```

---

## Dataset Description (26 columns)
| Column | Description |
|--------|-------------|
| `no_show` | **Target** — "yes" = patient missed appointment |
| `specialty` | Medical specialty (psychotherapy, physiotherapy, etc.) |
| `gender` | M / F |
| `age` | Patient age (21% missing → filled with median) |
| `disability` | intellectual / motor / unknown |
| `place` | City of residence |
| `appointment_shift` | morning / afternoon |
| `appointment_time` | Hour of appointment |
| `Hipertension/Diabetes/Alcoholism/Handcap/Scholarship` | Binary health flags |
| `SMS_received` | Whether reminder SMS was sent |
| `heat_intensity` / `rain_intensity` | Weather conditions |
| `average_temp_day` / `average_rain_day` | Numeric weather |
| `appointment_date_continuous` | Full date (no gaps) |

**Missing values:** age (~21%), specialty (~18%), disability (~15%), place (~10.5%)

---

## Model Performance

### Classification (No-Show Prediction)
| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.436 | 0.619 |
| Decision Tree | 0.568 | 0.720 |
| Random Forest | 0.535 | 0.764 |
| XGBoost | 0.545 | 0.774 |
| **LightGBM** | **0.540** | **0.775** |

Class imbalance handled with **SMOTE oversampling**.

### Demand Forecasting
| Model | R² |
|-------|----|
| Linear Regression | 0.152 |
| **Random Forest** | **0.194** |
| LightGBM | 0.037 |

Uses lag-1, lag-7, lag-14 and rolling averages as features.

---

## App Pages
1. **Dashboard** — KPIs, no-show rates by segment, time series view
2. **No-Show Predictor** — Input patient details → get risk score + recommendation
3. **Demand Forecaster** — Select date range → get daily appointment volume forecast
4. **Model Insights** — Feature importance, intervention strategies, model comparison

---

## Key Findings
- Overall no-show rate: **31.8%** (higher than typical 10-20%)
- **Appointment month** is the strongest predictor (May, July highest no-show)
- **Patient contact** (SMS/call) significantly reduces no-show risk
- **Cold weather months** correlate with higher absenteeism
- **Female patients** have slightly higher no-show rate than male
