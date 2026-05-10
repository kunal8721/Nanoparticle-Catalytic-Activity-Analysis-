import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- OUTLIER FUNCTION ----------
def remove_outliers(df):
    cols = [
        'size_nm','precursor_conc_mM','temp_C',
        'surface_area_m2_g','num_peaks_total',
        'peak_range','max_peak_intensity'
    ]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask &= (df[col] >= lower) & (df[col] <= upper)
    return df[mask]

# ---------- LOAD + TRAIN ----------
@st.cache_data
def load_and_train():
    df = pd.read_csv('nanomaterial_dataset_5000.csv')

    target = 'catalytic_activity'
    X_original = df.drop(columns=[target])
    X = X_original.copy()
    y = df[target]

    X_clean = remove_outliers(X)
    y_clean = y.loc[X_clean.index]

    X_clean = pd.get_dummies(X_clean, columns=[
        'element', 'material', 'synthesis_type'
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_clean, test_size=0.2, random_state=42
    )

    # -------- MODEL 1: LINEAR REGRESSION --------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # -------- MODEL 2: RANDOM FOREST --------
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # -------- MODEL 3: XGBOOST --------
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # -------- SCORES --------
    lr_score = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_rmse = np.sqrt(lr_mse)
    rf_score = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    xgb_score = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)

    # -------- SELECT BEST --------
    scores = {
        "Linear Regression": lr_score,
        "Random Forest": rf_score,
        "XGBoost": xgb_score
    }

    selected = max(scores, key=scores.get)

    if selected == "Linear Regression":
        model = lr_model
        R2 = lr_score
        MAE = lr_mae
        RMSE = lr_rmse
    elif selected == "Random Forest":
        model = rf_model
        R2 = rf_score
        MAE = rf_mae
        RMSE = rf_rmse
    else:
        model = xgb_model
        R2 = xgb_score
        MAE = xgb_mae
        RMSE = xgb_rmse
    return model, scaler, X_original,X_clean, selected, R2,MAE,RMSE,lr_score,rf_score,xgb_score,lr_mae,lr_rmse,rf_mae,rf_rmse,xgb_mae,xgb_rmse

# LOAD MODEL
model, scaler, X_original, X_clean, selected_model_name, R2, MAE, RMSE, lr_score, rf_score, xgb_score, lr_mae, lr_rmse, rf_mae, rf_rmse, xgb_mae, xgb_rmse = load_and_train()

# ---------- UI ----------
st.title("Catalytic Activity Predictor")
tab1,tab2,tab3 = st.tabs(["Model Analysis","DATA ANALYSIS", "Prediction"])
# -------- DATA ANALYSIS --------
with tab2:
    st.subheader("Outlier Removal Impact (Box Plot)")

    feature = st.selectbox(
        "Select Feature",
        ['size_nm','precursor_conc_mM','temp_C','surface_area_m2_g']
    )

    fig, ax = plt.subplots()

    ax.boxplot(
        [X_original[feature], X_clean[feature]],
        labels=["Before", "After"]
    )

    ax.set_title(f"{feature} Before vs After Outlier Removal")

    st.pyplot(fig)

# -------- MODEL COMPARISON --------
with tab1:
    st.subheader("Model Comparison")

    models = ["Linear Regression", "Random Forest", "XGBoost"]

    # -------- R2 --------
    st.write("R2 Score")
    r2_scores = [lr_score, rf_score, xgb_score]

    fig1, ax1 = plt.subplots()
    ax1.bar(models, r2_scores)
    ax1.set_title("R2 Score Comparison")
    st.pyplot(fig1)

    # -------- MAE --------
    st.write("MAE (Lower is Better)")
    mae_scores = [lr_mae, rf_mae, xgb_mae]

    fig2, ax2 = plt.subplots()
    ax2.bar(models, mae_scores)
    ax2.set_title("MAE Comparison")
    st.pyplot(fig2)

    # -------- RMSE --------
    st.write("RMSE (Lower is Better)")
    rmse_scores = [lr_rmse, rf_rmse, xgb_rmse]

    fig3, ax3 = plt.subplots()
    ax3.bar(models, rmse_scores)
    ax3.set_title("RMSE Comparison")
    st.pyplot(fig3)
with tab3:
    st.subheader("SELECTED MODEL")
    st.write(f"Selected Model: {selected_model_name}")
    st.write(f"R2 SCORE: {R2:.3f}")
    st.write(f"MAE: {MAE:.3f}")
    st.write(f"RMSE: {RMSE:.3f}")
    st.header("Input Parameters")

    size = st.slider("Size (nm)", 1, 100, 50)
    precursor = st.slider("Precursor Conc (mM)", 1, 50, 10)
    capping = st.slider("Capping Agent (%)", 0, 10, 2)
    ph = st.slider("pH", 1, 14, 7)
    temp = st.slider("Temperature (°C)", 20, 200, 100)
    stir_time = st.slider("Stirring Time (min)", 10, 200, 60)
    rpm = st.slider("RPM", 100, 1000, 300)
    surface = st.slider("Surface Area", 10, 300, 150)

    oh = st.slider("OH Count", 0, 5, 2)
    nh = st.slider("NH Count", 0, 5, 1)
    ch = st.slider("CH Count", 0, 5, 3)
    co = st.slider("C=O Count", 0, 5, 1)
    cc = st.slider("C=C Count", 0, 5, 2)
    co_st = st.slider("C-O Count", 0, 5, 2)
    metal_o = st.slider("Metal-O Count", 0, 5, 3)
    finger = st.slider("Fingerprint Count", 0, 10, 5)

    peaks = st.slider("Total Peaks", 5, 50, 20)
    pmin = st.slider("Peak Min", 200, 1000, 400)
    pmax = st.slider("Peak Max", 1000, 4000, 3500)
    prange = pmax - pmin
    mean = st.slider("Mean Wavenumber", 500, 3000, 1800)
    ppk = st.slider("Peaks per 1000", 1, 10, 6)
    density = st.slider("Peak Density", 0.1, 1.0, 0.5)
    intensity = st.slider("Max Intensity", 100, 2000, 900)

    element = st.selectbox("Element", ["Au", "Ag", "Cu"])
    material = st.selectbox("Material", ["oxide", "alloy"])
    synthesis = st.selectbox("Synthesis", ["sol-gel", "hydrothermal"])

    # ---------- PREDICTION ----------
    if st.button("Predict"):

        new_data = {
            'size_nm': size,
            'precursor_conc_mM': precursor,
            'capping_agent_pct': capping,
            'pH': ph,
            'temp_C': temp,
            'stirring_time_min': stir_time,
            'agitation_rpm': rpm,
            'surface_area_m2_g': surface,

            'OH_stretch_count': oh,
            'NH_stretch_count': nh,
            'CH_stretch_count': ch,
            'C=O_stretch_count': co,
            'C=C_aromatic_count': cc,
            'C-O_stretch_count': co_st,
            'Metal_O_bond_count': metal_o,
            'Fingerprint_region_count': finger,

            'num_peaks_total': peaks,
            'peak_min': pmin,
            'peak_max': pmax,
            'peak_range': prange,
            'mean_wavenumber': mean,
            'peaks_per_1000': ppk,
            'peak_density': density,
            'max_peak_intensity': intensity,

            'element': element,
            'material': material,
            'synthesis_type': synthesis
        }

        new_df = pd.DataFrame([new_data])

        new_df = pd.get_dummies(new_df)
        new_df = new_df.reindex(columns=X_clean.columns, fill_value=0)

        new_scaled = scaler.transform(new_df)

        pred = model.predict(new_scaled)[0]

        st.success(f"Predicted Activity: {pred:.3f}")

        if pred > 0.7:
            st.write("Reactivity: HIGH")
        elif pred > 0.4:
            st.write("Reactivity: MEDIUM")
        else:
            st.write("Reactivity: LOW")