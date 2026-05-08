"""
Medical Appointment No-Show Prediction & Demand Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "../../Medical_Appointment/Models" 

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        color: white; padding: 22px 28px; border-radius: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        border-left: 5px solid #2d6a9f;
    }
    .risk-high   { background:#fee2e2; border-left:5px solid #dc2626; border-radius:8px; padding:14px 18px; }
    .risk-medium { background:#fef9c3; border-left:5px solid #ca8a04; border-radius:8px; padding:14px 18px; }
    .risk-low    { background:#dcfce7; border-left:5px solid #16a34a; border-radius:8px; padding:14px 18px; }
    .insight-box { background:#eff6ff; border-radius:8px; padding:14px 18px; border-left:4px solid #3b82f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    clf = joblib.load(f"{MODELS_DIR}/best_classifier.pkl")
    clf_name = joblib.load(f"{MODELS_DIR}/best_classifier_name.pkl")
    forecaster = joblib.load(f"{MODELS_DIR}/demand_forecaster.pkl")
    le_dict = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
    features = joblib.load(f"{MODELS_DIR}/features.pkl")
    ts_features = joblib.load(f"{MODELS_DIR}/ts_features.pkl")
    cat_options = joblib.load(f"{MODELS_DIR}/cat_options.pkl")
    age_median = joblib.load(f"{MODELS_DIR}/age_median.pkl")
    last_values = joblib.load(f"{MODELS_DIR}/last_values.pkl")
    daily_ts = joblib.load(f"{MODELS_DIR}/daily_ts.pkl")
    return (clf, clf_name, forecaster, le_dict, features,
            ts_features, cat_options, age_median, last_values, daily_ts)


@st.cache_data
def load_raw_data():
    df = pd.read_csv("../../Medical_Appointment/Medical_appointment_data.csv")
    df["appointment_date"] = pd.to_datetime(df["appointment_date_continuous"])
    df["month"] = df["appointment_date"].dt.month
    df["day_of_week"] = df["appointment_date"].dt.dayofweek
    df["target"] = (df["no_show"] == "yes").astype(int)
    df["specialty"] = df["specialty"].fillna("unknown")
    df["place"] = df["place"].fillna("unknown")
    return df


# Load everything
(clf, clf_name, forecaster, le_dict, FEATURES,
 TS_FEATURES, cat_options, age_median, last_values, daily_ts) = load_artefacts()
df_raw = load_raw_data()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CER Medical Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["Dashboard", "No-Show Predictor", "Demand Forecaster", "Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"""
        <div style='font-size:12px; color:#666'>
        <b>Dataset:</b> 109,593 appointments<br>
        <b>Best Classifier:</b> {clf_name}<br>
        <b>No-Show Rate:</b> {(df_raw['no_show']=='yes').mean()*100:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown(
        """
        <div class='main-header'>
          <h2 style='margin:0'>Operations Dashboard</h2>
          <p style='margin:4px 0 0; opacity:0.85'>
            Medical Appointment Analytics Overview
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row
    total = len(df_raw)
    noshow_n = (df_raw["no_show"] == "yes").sum()
    noshow_rate = noshow_n / total * 100
    show_n = total - noshow_n
    sms_show = (df_raw[df_raw.SMS_received == 1]["no_show"] == "no").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Appointments", f"{total:,}")
    c2.metric("Attended", f"{show_n:,}", f"{show_n/total*100:.1f}%")
    c3.metric("No-Show", f"{noshow_n:,}", f"-{noshow_rate:.1f}%")
    c4.metric("SMS Attendance Rate", f"{sms_show:.1f}%")

    st.markdown("---")

    # Row 1: Gender + Specialty
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No-Show Rate by Gender")
        grp = (df_raw.groupby("gender")["target"].mean() * 100).reset_index()
        grp = grp[grp['gender'].isin(['M','F'])]
        fig_gender = px.bar(
            grp, x="gender", y="target", text_auto='.1f',
            color="gender", color_discrete_sequence=["#4C72B0", "#DD8452"],
            labels={"target": "No-Show Rate (%)", "gender": "Gender"}
        )
        fig_gender.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        st.subheader("No-Show Rate by Specialty")
        spec = (df_raw.groupby("specialty")["target"].mean() * 100)
        spec = spec[spec.index != "unknown"].sort_values(ascending=True).reset_index()
        fig_spec = px.bar(
            spec, x="target", y="specialty", orientation="h",
            color="target", color_continuous_scale="Blues",
            labels={"target": "No-Show Rate (%)", "specialty": "Specialty"}
        )
        fig_spec.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_spec, use_container_width=True)

    # Row 2: Monthly + Weather
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("No-Show Rate by Month")
        month_ns = (df_raw.groupby("month")["target"].mean() * 100).reset_index()
        fig_month = px.area(
            month_ns, x="month", y="target", markers=True,
            color_discrete_sequence=["#2d6a9f"],
            labels={"target": "No-Show Rate (%)", "month": "Month"}
        )
        fig_month.update_xaxes(tickmode='linear')
        fig_month.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_month, use_container_width=True)

    with col4:
        st.subheader("No-Show Rate by Heat Intensity")
        heat_order = ["heavy_cold", "cold", "mild", "warm", "heavy_warm"]
        heat_ns = (df_raw.groupby("heat_intensity")["target"].mean() * 100)
        heat_ns = heat_ns.reindex([h for h in heat_order if h in heat_ns.index]).reset_index()
        fig_heat = px.bar(
            heat_ns, x="heat_intensity", y="target", text_auto='.1f',
            color_discrete_sequence=["#4C72B0"],
            labels={"target": "No-Show Rate (%)", "heat_intensity": "Heat Intensity"}
        )
        fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Row 3: Daily Time Series
    st.subheader("Daily Appointment Volume Over Time")
    daily_ts_copy = daily_ts.copy()
    daily_ts_copy["appointment_date"] = pd.to_datetime(daily_ts_copy["appointment_date"])
    daily_ts_copy["rolling_14"] = daily_ts_copy["demand"].rolling(14, min_periods=1).mean()
    
    fig_ts = px.line(
        daily_ts_copy, x="appointment_date", y=["demand", "rolling_14"],
        labels={"value": "Appointments", "appointment_date": "Date", "variable": "Legend"},
        color_discrete_map={"demand": "rgba(76, 114, 176, 0.4)", "rolling_14": "#DD8452"}
    )
    # Style the rolling average line to be thicker
    fig_ts.data[1].line.width = 3
    fig_ts.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ts, use_container_width=True)

    # Health conditions
    st.subheader("Health Condition Impact")
    health_cols = ["Hipertension", "Diabetes", "Alcoholism", "Handcap", "Scholarship", "SMS_received"]
    rows = []
    for col in health_cols:
        no = (df_raw[df_raw[col] == 0]["target"].mean() * 100)
        yes = (df_raw[df_raw[col] == 1]["target"].mean() * 100)
        rows.append({"Condition": col, "No-Show % (without)": round(no, 1), "No-Show % (with)": round(yes, 1)})
    impact_df = pd.DataFrame(rows).set_index("Condition")
    st.dataframe(impact_df.style.background_gradient(cmap="RdYlGn_r", axis=None), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — NO-SHOW PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "No-Show Predictor":
    st.markdown(
        """
        <div class='main-header'>
          <h2 style='margin:0'>No-Show Risk Predictor</h2>
          <p style='margin:4px 0 0; opacity:0.85'>
            Enter patient & appointment details to get a no-show probability score
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(f"**Model:** {clf_name}  |  **Features:** {len(FEATURES)}  |  Trained on 109,593 appointments with SMOTE oversampling.")

    with st.form("predict_form"):
        st.subheader("Patient Information")
        p1, p2, p3 = st.columns(3)
        with p1:
            gender = st.selectbox("Gender", cat_options.get("gender", ["F", "M"]))
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
        with p2:
            disability = st.selectbox("Disability Type", cat_options.get("disability", ["intellectual", "motor", "unknown"]))
            needs_companion = st.selectbox("Needs Companion", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with p3:
            under_12 = st.selectbox("Under 12 Years Old", [0, 1], format_func=lambda x: "Yes" if x else "No")
            over_60 = st.selectbox("Over 60 Years Old", [0, 1], format_func=lambda x: "Yes" if x else "No")

        st.subheader("Appointment Details")
        a1, a2, a3 = st.columns(3)
        with a1:
            specialty = st.selectbox("Specialty", cat_options.get("specialty", []))
            shift = st.selectbox("Appointment Shift", cat_options.get("appointment_shift", ["morning", "afternoon"]))
        with a2:
            appt_time = st.slider("Appointment Hour (24h)", 7, 18, 10)
            appt_month = st.selectbox("Appointment Month", list(range(1, 13)))
        with a3:
            appt_dow = st.selectbox("Day of Week", list(range(7)),
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            place = st.selectbox("City / Place", cat_options.get("place", []))

        st.subheader("Weather & Environment")
        w1, w2, w3 = st.columns(3)
        with w1:
            heat_intensity = st.selectbox("Heat Intensity",
                cat_options.get("heat_intensity", ["heavy_cold","cold","mild","warm","heavy_warm"]))
            rain_intensity = st.selectbox("Rain Intensity",
                cat_options.get("rain_intensity", ["no_rain","weak","moderate","heavy"]))
        with w2:
            avg_temp = st.slider("Average Temperature (°C)", 10.0, 40.0, 22.0, step=0.5)
            max_temp = st.slider("Max Temperature (°C)", 10.0, 45.0, 28.0, step=0.5)
        with w3:
            rainy_before = st.selectbox("Rainy Day Before", [0, 1], format_func=lambda x: "Yes" if x else "No")
            storm_before = st.selectbox("Storm Day Before", [0, 1], format_func=lambda x: "Yes" if x else "No")
            avg_rain = st.slider("Average Rainfall (mm)", 0.0, 60.0, 0.0, step=0.5)
            
        st.subheader("Health Conditions")
        h1, h2, h3 = st.columns(3)
        with h1:
            hipert = st.checkbox("Hypertension"); diabetes = st.checkbox("Diabetes")
        with h2:
            alcohol = st.checkbox("Alcoholism"); handcap = st.checkbox("Disability")
        with h3:
            scholar = st.checkbox("Scholarship (Bolsa Família)"); sms_recv = st.checkbox("SMS Reminder Sent")

        submitted = st.form_submit_button("Predict No-Show Risk", use_container_width=True)

    if submitted:
        with st.spinner("Running prediction…"):
            def safe_encode(le, val):
                classes = list(le.classes_)
                if val in classes:
                    return le.transform([val])[0]
                return 0  # fallback

            row = {
                "gender_enc": safe_encode(le_dict["gender"], gender),
                "specialty_enc": safe_encode(le_dict["specialty"], specialty),
                "disability_enc": safe_encode(le_dict["disability"], disability),
                "place_enc": safe_encode(le_dict["place"], place),
                "appointment_shift_enc": safe_encode(le_dict["appointment_shift"], shift),
                "heat_intensity_enc": safe_encode(le_dict["heat_intensity"], heat_intensity),
                "rain_intensity_enc": safe_encode(le_dict["rain_intensity"], rain_intensity),
                "age": age,
                "appointment_time": appt_time,
                "month": appt_month,
                "day_of_week": appt_dow,
                "week_of_year": int(pd.Timestamp(f"2020-{appt_month:02d}-15").strftime("%V")),
                "quarter": (appt_month - 1) // 3 + 1,
                "is_weekend": 1 if appt_dow >= 5 else 0,
                "under_12_years_old": under_12,
                "over_60_years_old": over_60,
                "patient_needs_companion": needs_companion,
                "average_temp_day": avg_temp,
                "average_rain_day": avg_rain,
                "max_temp_day": max_temp,
                "max_rain_day": avg_rain,
                "rainy_day_before": rainy_before,
                "storm_day_before": storm_before,
                "Hypertension": int(hipert),
                "Diabetes": int(diabetes),
                "Alcoholism": int(alcohol),
                "Handcap": int(handcap),
                "Scholarship": int(scholar),
                "SMS_received": int(sms_recv),
            }

            X_input = pd.DataFrame([row])[FEATURES]
            prob = clf.predict_proba(X_input)[0, 1]
            pred = clf.predict(X_input)[0]

        # Risk display
        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1, 1])
        r1.metric("No-Show Probability", f"{prob*100:.1f}%")
        r2.metric("Predicted Outcome", "No-Show" if pred == 1 else "Will Attend")
        r3.metric("Risk Level", "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW"))

        # Gauge Chart using Plotly
        gauge_color = "#dc2626" if prob > 0.6 else ("#ca8a04" if prob > 0.35 else "#16a34a")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'valueformat': ".1f"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 35], 'color': '#dcfce7'},
                    {'range': [35, 60], 'color': '#fef9c3'},
                    {'range': [60, 100], 'color': '#fee2e2'}],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': prob * 100}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendation
        if prob > 0.6:
            st.markdown(
                f"<div class='risk-high'><b>HIGH RISK — Immediate Action Required</b><br>"
                f"Probability: <b>{prob*100:.1f}%</b>. Send SMS reminder, consider a call, or enable overbooking for this slot.</div>",
                unsafe_allow_html=True,
            )
        elif prob > 0.35:
            st.markdown(
                f"<div class='risk-medium'><b>MEDIUM RISK — Monitor This Appointment</b><br>"
                f"Probability: <b>{prob*100:.1f}%</b>. Send a reminder SMS and flag this slot for possible re-allocation.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='risk-low'><b>LOW RISK — Patient Likely to Attend</b><br>"
                f"Probability: <b>{prob*100:.1f}%</b>. Standard scheduling applies.</div>",
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEMAND FORECASTER
# ════════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecaster":
    st.markdown(
        """
        <div class='main-header'>
          <h2 style='margin:0'>Appointment Demand Forecaster</h2>
          <p style='margin:4px 0 0; opacity:0.85'>
            Predict daily appointment volume for future planning & staff scheduling
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("**How it works:** Enter a future date range and appointment context. The model predicts daily appointment load using temporal patterns learned from historical data.")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Forecast Settings")
        start_date = st.date_input("Start Date", value=date(2021, 6, 1))
        end_date = st.date_input("End Date", value=date(2021, 6, 14))

        if end_date <= start_date:
            st.error("End date must be after start date.")
            st.stop()

        run_forecast = st.button("Generate Forecast", use_container_width=True)

    with col_r:
        st.subheader("Historical Demand (Last 90 Days)")
        daily_ts_copy = daily_ts.copy()
        daily_ts_copy["appointment_date"] = pd.to_datetime(daily_ts_copy["appointment_date"])
        recent = daily_ts_copy.tail(90)
        
        fig_recent = px.area(
            recent, x="appointment_date", y="demand",
            color_discrete_sequence=["#4C72B0"],
            labels={"demand": "Daily Appointments", "appointment_date": "Date"}
        )
        fig_recent.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
        st.plotly_chart(fig_recent, use_container_width=True)

    if run_forecast:
        with st.spinner("Generating forecast…"):
            date_range = pd.date_range(start_date, end_date)
            lv = last_values  # last known lag values from training data

            forecast_rows = []
            rolling_window = [lv["lag_7"], lv["lag_7"], lv["lag_7"],
                              lv["lag_7"], lv["lag_7"], lv["lag_7"],
                              lv["lag_1"]]  # simple fill
            recent_vals = list(daily_ts_copy["demand"].iloc[-14:].values)

            for i, d in enumerate(date_range):
                lag1 = recent_vals[-1] if recent_vals else lv["lag_1"]
                lag7 = recent_vals[-7] if len(recent_vals) >= 7 else lv["lag_7"]
                lag14 = recent_vals[-14] if len(recent_vals) >= 14 else lv["lag_14"]
                roll7 = np.mean(recent_vals[-7:]) if len(recent_vals) >= 7 else lv["rolling_7"]
                roll14 = np.mean(recent_vals[-14:]) if len(recent_vals) >= 14 else lv["rolling_14"]

                row = {
                    "day_of_week": d.dayofweek,
                    "month": d.month,
                    "week_of_year": int(d.strftime("%V")),
                    "quarter": (d.month - 1) // 3 + 1,
                    "day_of_month": d.day,
                    "is_weekend": int(d.dayofweek >= 5),
                    "is_month_start": int(d.is_month_start),
                    "is_month_end": int(d.is_month_end),
                    "lag_1": lag1, "lag_7": lag7, "lag_14": lag14,
                    "rolling_7": roll7, "rolling_14": roll14,
                }
                X_fc = pd.DataFrame([row])[TS_FEATURES]
                pred_demand = max(0, float(forecaster.predict(X_fc)[0]))
                forecast_rows.append({
                    "Date": d.strftime("%Y-%m-%d"),
                    "Day": d.strftime("%A"),
                    "Predicted Demand": round(pred_demand),
                    "Day Type": "Weekend" if row["is_weekend"] else "Weekday",
                })
                recent_vals.append(pred_demand)

        fc_df = pd.DataFrame(forecast_rows)

        # Show forecast table
        st.subheader("Forecast Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Forecast Period", f"{len(fc_df)} days")
        c2.metric("Avg Daily Demand", f"{fc_df['Predicted Demand'].mean():.0f}")
        c3.metric("Peak Day", fc_df.loc[fc_df["Predicted Demand"].idxmax(), "Date"])

        # Plot forecast
        fig_fc = px.bar(
            fc_df, x="Date", y="Predicted Demand", color="Day Type",
            color_discrete_map={"Weekday": "#4C72B0", "Weekend": "#DD8452"},
            labels={"Predicted Demand": "Predicted Appointments"}
        )
        fig_fc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Table
        display_df = fc_df[["Date", "Day", "Predicted Demand"]].copy()
        display_df["Staffing Recommendation"] = display_df["Predicted Demand"].apply(
            lambda x: "Full Staff" if x > 300 else ("Standard" if x > 100 else "Minimal")
        )
        st.dataframe(display_df.style.background_gradient(
            subset=["Predicted Demand"], cmap="Blues"), use_container_width=True)

        # Weekly demand summary
        st.subheader("Historical Weekly Demand Trend")
        daily_ts_copy["week"] = daily_ts_copy["appointment_date"].dt.isocalendar().week.astype(int)
        daily_ts_copy["year"] = daily_ts_copy["appointment_date"].dt.year
        weekly = daily_ts_copy.groupby(["year","week"])["demand"].sum().reset_index()
        weekly["period"] = weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2)
        
        fig_weekly = px.area(
            weekly, x="period", y="demand",
            color_discrete_sequence=["#2d6a9f"],
            labels={"demand": "Weekly Appointments", "period": "Time Period"}
        )
        fig_weekly.update_xaxes(tickangle=45)
        fig_weekly.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_weekly, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.markdown(
        """
        <div class='main-header'>
          <h2 style='margin:0'>Model Insights & Feature Importance</h2>
          <p style='margin:4px 0 0; opacity:0.85'>
            Deep-dive into model performance, feature importance, and intervention strategies
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Classifier", "Forecaster", "Interventions"])

    with tab1:
        st.subheader(f"Best Classifier: {clf_name}")

        m1, m2, m3 = st.columns(3)
        m1.markdown("<div class='metric-card'><h4>Model</h4><h2>{}</h2></div>".format(clf_name), unsafe_allow_html=True)
        m2.markdown("<div class='metric-card'><h4>Training Strategy</h4><h2>SMOTE + Stratified</h2></div>", unsafe_allow_html=True)
        m3.markdown("<div class='metric-card'><h4>Validation</h4><h2>Hold-out 30%</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Feature importance
        if hasattr(clf, "feature_importances_"):
            fi = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(20)
            fi_df = fi.reset_index()
            fi_df.columns = ["Feature", "Importance Score"]
            
            fig_fi = px.bar(
                fi_df, x="Importance Score", y="Feature", orientation="h",
                color="Importance Score", color_continuous_scale="Blues",
                title=f"Top 20 Feature Importances — {clf_name}"
            )
            fig_fi.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=500)
            st.plotly_chart(fig_fi, use_container_width=True)

            st.subheader("Top 5 Most Important Features")
            cols = st.columns(5)
            for i, (feat, val) in enumerate(fi.sort_values(ascending=False).head(5).items()):
                cols[i].metric(feat.replace("_enc","").replace("_"," ").title(), f"{val:.4f}")
        else:
            st.info("Feature importance not available for this model type.")

        # Model description
        st.markdown("---")
        st.subheader("About the Models Compared")
        model_desc = {
            "Logistic Regression": "Linear model — interpretable baseline. Works well for linearly separable data but limited for complex patterns.",
            "Decision Tree": "Simple tree-based model with max_depth=6. Easy to interpret, prone to overfitting without constraints.",
            "Random Forest": "Ensemble of 100 decision trees. Robust, handles non-linearity, good for tabular data.",
            "XGBoost": "Gradient boosted trees with regularization. State-of-the-art for tabular data in competitions.",
            "LightGBM": "Fast gradient boosting framework by Microsoft. Efficient on large datasets, often best AUC."
        }
        for name, desc in model_desc.items():
            st.markdown(f"**{name}:** {desc}")

    with tab2:
        st.subheader("Demand Forecasting Model")
        st.markdown("""
        **Approach:** Supervised regression using temporal features engineered from appointment dates.

        **Feature Engineering:**
        - Lag features: demand 1, 7, and 14 days prior
        - Rolling averages: 7-day and 14-day windows
        - Calendar features: day of week, month, quarter, week of year
        - Binary flags: is_weekend, is_month_start, is_month_end

        **Train/Test Split:** Chronological 70/30 — no data leakage from future to past.
        """)

        if hasattr(forecaster, "feature_importances_"):
            fi_reg = pd.Series(forecaster.feature_importances_, index=TS_FEATURES).sort_values(ascending=True)
            fi_reg_df = fi_reg.reset_index()
            fi_reg_df.columns = ["Feature", "Importance Score"]
            
            fig_reg_fi = px.bar(
                fi_reg_df, x="Importance Score", y="Feature", orientation="h",
                color_discrete_sequence=["#4C72B0"],
                title="Demand Forecast — Feature Importances"
            )
            fig_reg_fi.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=400)
            st.plotly_chart(fig_reg_fi, use_container_width=True)

        st.markdown("""
        > **Note on Forecast Accuracy:** The dataset spans multiple years with significant 
        > structural shifts (e.g., COVID-19 impact in 2020-2021). This causes high variance 
        > in daily demand, limiting R² on test data. In production, retrain on the most 
        > recent 6-12 months for best accuracy.
        """)

    with tab3:
        st.subheader("Actionable Intervention Strategies")

        strategies = [
            {
                "title": "Targeted SMS Reminders",
                "detail": "Send SMS to patients with predicted no-show probability > 35%. "
                          "Analysis shows SMS recipients have statistically different attendance patterns.",
                "impact": "High",
            },
            {
                "title": "Smart Overbooking",
                "detail": "For slots with multiple high-risk patients (avg probability > 50%), "
                          "consider booking 1 additional patient to compensate for expected no-shows.",
                "impact": "High",
            },
            {
                "title": "Proactive Call Campaign",
                "detail": "For patients with probability > 70%, escalate to phone call 48h before appointment.",
                "impact": "Medium",
            },
            {
                "title": "Weather-Aware Scheduling",
                "detail": "During predicted cold or stormy days (heavy_cold / storm_day_before), "
                          "reduce non-urgent appointment load and increase buffer slots.",
                "impact": "Medium",
            },
            {
                "title": "Month-Based Capacity Planning",
                "detail": "No-show rates peak in May, June, and July. Schedule extra capacity "
                          "or implement waitlist systems during these months.",
                "impact": "Medium",
            },
            {
                "title": "Under-12 Companion Policy",
                "detail": "Minor patients require companions — ensure reminder messages include "
                          "companion logistics to reduce attendance barriers.",
                "impact": "Low-Medium",
            },
        ]

        for s in strategies:
            color = "#fee2e2" if s["impact"] == "High" else ("#fef9c3" if s["impact"] == "Medium" else "#eff6ff")
            st.markdown(
                f"<div style='background:{color}; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>"
                f"<b>{s['title']}</b> — Impact: <b>{s['impact']}</b><br>{s['detail']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("No-Show Rate Summary by Key Segments")
        summary_data = {
            "Segment": ["Gender: Female", "Gender: Male", "SMS Received", "No SMS", "Shift: Morning", "Shift: Afternoon"],
            "No-Show Rate (%)": [
                round((df_raw[df_raw.gender=="F"]["target"].mean())*100, 1),
                round((df_raw[df_raw.gender=="M"]["target"].mean())*100, 1),
                round((df_raw[df_raw.SMS_received==1]["target"].mean())*100, 1),
                round((df_raw[df_raw.SMS_received==0]["target"].mean())*100, 1),
                round((df_raw[df_raw.appointment_shift=="morning"]["target"].mean())*100, 1),
                round((df_raw[df_raw.appointment_shift=="afternoon"]["target"].mean())*100, 1),
            ]
        }
        st.dataframe(pd.DataFrame(summary_data).style.background_gradient(
            subset=["No-Show Rate (%)"], cmap="RdYlGn_r"), use_container_width=True)