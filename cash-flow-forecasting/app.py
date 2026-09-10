
# app.py — Business Cash Flow Forecasting + QR Share (Dashboard style)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import qrcode
from io import BytesIO

# ----------------------------------------------------
# Streamlit Page Setup
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="💼 Business Cash Flow Forecasting Dashboard")

# ----------------------------------------------------
# Dashboard Header (CSS + Title)
# ----------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            color: #003366;
        }
        .metric-container {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
            padding: 12px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Business Cash Flow Forecasting Dashboard")
st.markdown("### 📊 Financial Overview & Forecasting Insights")
st.markdown("---")

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
def read_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names, detects 'date', 'inflow', 'outflow' columns,
    coerces date, aggregates to monthly, and computes net_cash.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # detect date column
    if 'date' not in df.columns:
        detected = [c for c in df.columns if 'date' in c]
        if detected:
            df = df.rename(columns={detected[0]: 'date'})
        else:
            possible_names = ['transaction date', 'dates', 'day', 'month', 'period']
            for name in possible_names:
                if name in df.columns:
                    df = df.rename(columns={name: 'date'})
                    break
            else:
                st.error(f"❌ No 'date' column found. Columns found: {list(df.columns)}")
                st.stop()

    # rename common inflow/outflow names
    rename_map = {
        'cash in': 'inflow', 'cash_in': 'inflow', 'income': 'inflow', 'sales': 'inflow',
        'cash out': 'outflow', 'cash_out': 'outflow', 'expense': 'outflow', 'spend': 'outflow',
        'transaction date': 'date', 'dates': 'date', 'day': 'date', 'month': 'date'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # final check
    if not {'date'}.issubset(df.columns):
        st.error("❌ Missing required column: date.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    if 'inflow' not in df.columns:
        df['inflow'] = 0.0
    if 'outflow' not in df.columns:
        df['outflow'] = 0.0

    df['inflow'] = pd.to_numeric(df['inflow'], errors='coerce').fillna(0.0)
    df['outflow'] = pd.to_numeric(df['outflow'], errors='coerce').fillna(0.0)

    # monthly aggregation
    df_month = df.set_index('date').resample('M').sum(numeric_only=True).reset_index()
    df_month['net_cash'] = df_month['inflow'] - df_month['outflow']
    return df_month

def make_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f'net_lag_{lag}'] = df['net_cash'].shift(lag)
    df['month'] = df['date'].dt.month
    df = df.dropna().reset_index(drop=True)
    return df

def train_random_forest(df_features: pd.DataFrame):
    X = df_features.drop(columns=['date', 'inflow', 'outflow', 'net_cash'], errors='ignore')
    y = df_features['net_cash']
    if len(X) < 5:
        st.warning("Not enough rows after lagging to train Random Forest. Need at least 5 rows.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds) if len(y_test) > 0 else float('nan')
    return model, mae

def forecast_rf(model, recent_df: pd.DataFrame, n_periods: int = 3):
    results = []
    current = recent_df.copy().reset_index(drop=True)
    if len(current) < 3:
        raise ValueError("Need at least 3 months of recent data to make RF forecast (for 3 lags).")
    for _ in range(n_periods):
        last_date = current['date'].max()
        next_date = last_date + pd.DateOffset(months=1)
        next_month = next_date.month
        lags = [current['net_cash'].iloc[-lag] for lag in range(1, 4)]
        feature = {'net_lag_1': lags[0], 'net_lag_2': lags[1], 'net_lag_3': lags[2], 'month': next_month}
        X_pred = pd.DataFrame([feature])
        pred_net = float(model.predict(X_pred)[0])
        last_inflow = float(current['inflow'].iloc[-1])
        predicted_inflow = last_inflow + max(pred_net, 0) * 0.6
        predicted_outflow = predicted_inflow - pred_net
        results.append({
            'date': next_date,
            'predicted_inflow': round(predicted_inflow, 2),
            'predicted_outflow': round(predicted_outflow, 2),
            'predicted_net_cash': round(pred_net, 2)
        })
        new_row = {
            'date': next_date,
            'inflow': predicted_inflow,
            'outflow': predicted_outflow,
            'net_cash': pred_net
        }
        current = pd.concat([current, pd.DataFrame([new_row])], ignore_index=True)
    return pd.DataFrame(results)

def train_arima(df: pd.DataFrame):
    series = df.set_index('date')['net_cash'].asfreq('M').fillna(method='ffill')
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()
    fitted = model_fit.fittedvalues
    min_len = min(len(series), len(fitted))
    if min_len >= 1:
        mae = mean_absolute_error(series[-min_len:], fitted[-min_len:])
    else:
        mae = float('nan')
    return model_fit, mae

def forecast_arima(model_fit, df: pd.DataFrame, n_periods: int):
    forecast = model_fit.forecast(steps=n_periods)
    last_date = df['date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, n_periods + 1)]
    return pd.DataFrame({
        'date': future_dates,
        'predicted_net_cash': np.round(forecast, 2),
        'predicted_inflow': np.nan,
        'predicted_outflow': np.nan
    })

def train_prophet(df: pd.DataFrame):
    prophet_df = df[['date', 'net_cash']].rename(columns={'date': 'ds', 'net_cash': 'y'})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    if len(prophet_df) >= 6:
        y_true = prophet_df['y'][-6:].values
        y_pred = forecast['yhat'][:-3][-6:].values
        mae = mean_absolute_error(y_true, y_pred)
    else:
        mae = float('nan')
    return model, mae

def forecast_prophet(model: Prophet, df: pd.DataFrame, n_periods: int):
    future = model.make_future_dataframe(periods=n_periods, freq='M')
    forecast = model.predict(future)
    future = forecast[['ds', 'yhat']].tail(n_periods)
    future.columns = ['date', 'predicted_net_cash']
    return future

# ----------------------------------------------------
# Streamlit UI Inputs
# ----------------------------------------------------
uploaded = st.file_uploader("📂 Upload CSV (columns: date, inflow, outflow, category optional)", type=['csv'])

col1, col2, col3 = st.columns([1, 1, 1])
model_choice = col1.selectbox("Select Model", ["Random Forest", "ARIMA", "Prophet"])
period_select = col2.selectbox("Select Forecast Period (months)", [1, 2, 3, 6, 12], index=2)
predict_button = col3.button("🔮 Predict")

# ----------------------------------------------------
# Data Load (sample if no file)
# ----------------------------------------------------
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    st.info("No file uploaded — using generated sample dataset.")
    dates = pd.date_range(start="2023-01-01", end="2025-10-01", freq="MS")
    np.random.seed(42)
    inflow = np.random.randint(120000, 350000, len(dates))
    outflow = inflow - np.random.randint(20000, 80000, len(dates))
    categories = np.random.choice(['Sales', 'Marketing', 'Operations', 'Rent'], len(dates))
    df_raw = pd.DataFrame({'date': dates, 'inflow': inflow, 'outflow': outflow, 'category': categories})

# --- Fix: ensure a 'date' column exists ---
cols_lower = [c.lower().strip() for c in df_raw.columns]
df_raw.columns = cols_lower
if 'date' not in df_raw.columns:
    # try auto-detect common names
    for c in ['transaction date', 'dates', 'day', 'month', 'period']:
        if c in df_raw.columns:
            df_raw.rename(columns={c: 'date'}, inplace=True)
            break
if 'date' not in df_raw.columns:
    st.error(f"❌ No 'date' column found in uploaded CSV. Columns found: {list(df_raw.columns)}")
    st.stop()

df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')


st.sidebar.subheader("📄 Data Preview")
st.sidebar.dataframe(df_raw.head())

# ----------------------------------------------------
# Prepare baseline monthly aggregation (full dataset)
# ----------------------------------------------------
df_month_all = read_and_prepare(df_raw)

# ----------------------------------------------------
# Filters Section (Year + Category) - sidebar
# ----------------------------------------------------
st.sidebar.markdown("## 🔍 Filters")

years = sorted(df_month_all['date'].dt.year.unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

if 'category' in df_raw.columns:
    categories = sorted(df_raw['category'].dropna().unique())
    selected_categories = st.sidebar.multiselect("Select Category", categories, default=categories)
    df_filtered_raw = df_raw[
        (df_raw['date'].dt.year.isin(selected_years)) &
        (df_raw['category'].isin(selected_categories))
    ]
else:
    df_filtered_raw = df_raw[df_raw['date'].dt.year.isin(selected_years)]

# Re-prepare monthly from filtered raw
df_month = read_and_prepare(df_filtered_raw)

st.sidebar.success(f"Showing: {', '.join(map(str, selected_years))} "
                   f"{' | Categories: ' + ','.join(selected_categories) if 'selected_categories' in locals() else ''}")

# ----------------------------------------------------
# KPIs (based on filtered df_month)
# ----------------------------------------------------
total_inflow = int(df_month['inflow'].sum())
total_outflow = int(df_month['outflow'].sum())
net_cash_total = int(df_month['net_cash'].sum())

st.subheader("💰 Financial Performance Summary")
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("💸 Total Inflow", f"₹ {total_inflow:,}", delta=f"{(total_inflow/1_000_000):.2f}M")
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("📤 Total Outflow", f"₹ {total_outflow:,}", delta=f"{(total_outflow/1_000_000):.2f}M")
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("📈 Net Cash", f"₹ {net_cash_total:,}", delta=f"{(net_cash_total/1_000_000):.2f}M")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# Cash Reserve Ratio (CRR) + Gauge
# ----------------------------------------------------
st.subheader("🏦 Cash Reserve Ratio (CRR)")

try:
    cash_reserve_ratio = (df_month['net_cash'].sum() / df_month['inflow'].sum()) * 100
except Exception:
    cash_reserve_ratio = 0.0

st.markdown(f"**Cash Reserve Ratio:** {cash_reserve_ratio:.2f}%")

fig_crr = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=round(cash_reserve_ratio, 2),
    delta={'reference': 25, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
    title={'text': "Cash Reserve Ratio (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 25], 'color': 'red'},
            {'range': [25, 50], 'color': 'orange'},
            {'range': [50, 75], 'color': 'yellow'},
            {'range': [75, 100], 'color': 'green'}
        ],
        'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': cash_reserve_ratio}
    }
))
st.plotly_chart(fig_crr, use_container_width=True)

if cash_reserve_ratio >= 75:
    st.success("💹 Excellent Liquidity: Company maintains strong reserves.")
elif cash_reserve_ratio >= 50:
    st.info("💼 Stable Liquidity: Cash flow is balanced, but monitor outflows.")
elif cash_reserve_ratio >= 25:
    st.warning("⚠️ Low Liquidity: Company may face short-term cash strain.")
else:
    st.error("🚨 Critical Liquidity Issue: Immediate action required to control outflows.")

# ----------------------------------------------------
# Visual Dashboard (Tabs + Charts)
# ----------------------------------------------------
st.markdown("---")
st.subheader("📊 Financial Insights Dashboard")
tab1, tab2, tab3 = st.tabs(["💸 Cash Flow Overview", "📆 Trends & Averages", "🧩 Category Analysis"])

with tab1:
    st.write("#### Monthly Inflow vs Outflow")
    fig_bar = px.bar(df_month, x='date', y=['inflow', 'outflow'],
                     barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("#### Net Cash Over Time")
    fig_area = px.area(df_month, x='date', y='net_cash')
    st.plotly_chart(fig_area, use_container_width=True)

with tab2:
    st.write("#### 3-Month Rolling Average (Net Cash)")
    df_month['rolling_avg'] = df_month['net_cash'].rolling(window=3).mean()
    fig_roll = px.line(df_month, x='date', y='rolling_avg')
    st.plotly_chart(fig_roll, use_container_width=True)

    st.write("#### Correlation Between Variables")
    corr_df = df_month[['inflow', 'outflow', 'net_cash']].corr()
    fig_heat = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    if 'category' in df_filtered_raw.columns:
        st.write("#### Category-Wise Cash Flow Summary")
        cat_summary = df_filtered_raw.groupby('category')[['inflow', 'outflow']].sum().reset_index()
        cat_summary['net_cash'] = cat_summary['inflow'] - cat_summary['outflow']
        fig_pie = px.pie(cat_summary, values='net_cash', names='category', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_cat_bar = px.bar(cat_summary, x='category', y=['inflow', 'outflow', 'net_cash'], barmode='group')
        st.plotly_chart(fig_cat_bar, use_container_width=True)
    else:
        st.info("No category column available in the filtered data.")

# ----------------------------------------------------
# Model Training
# ----------------------------------------------------
model = None
mae = float('nan')
try:
    if model_choice == "Random Forest":
        df_feat = make_lag_features(df_month)
        model, mae = train_random_forest(df_feat)
    elif model_choice == "ARIMA":
        model, mae = train_arima(df_month)
    else:  # Prophet
        model, mae = train_prophet(df_month)

    st.success(f"✅ {model_choice} model trained — Validation MAE: {mae:.2f}")
except Exception as e:
    st.error(f"Model training failed: {e}")

# ----------------------------------------------------
# Forecasting
# ----------------------------------------------------
if predict_button:
    if model is None:
        st.error("Model not available to forecast. Train succeeded before predicting.")
    else:
        n = int(period_select)
        try:
            if model_choice == "Random Forest":
                recent = df_month.tail(3).reset_index(drop=True)
                forecast_df = forecast_rf(model, recent, n_periods=n)
            elif model_choice == "ARIMA":
                forecast_df = forecast_arima(model, df_month, n_periods=n)
            else:
                forecast_df = forecast_prophet(model, df_month, n_periods=n)

            st.subheader("🔮 Predicted Cash Flow")
            forecast_df_display = forecast_df.copy()
            forecast_df_display['Month'] = forecast_df_display['date'].dt.strftime('%b %Y')

            if 'predicted_inflow' in forecast_df_display.columns:
                forecast_df_display = forecast_df_display[['Month', 'predicted_inflow', 'predicted_outflow', 'predicted_net_cash']]
                forecast_df_display.columns = ['Month', 'Predicted Inflow', 'Predicted Outflow', 'Predicted Net Cash']
            else:
                forecast_df_display = forecast_df_display[['Month', 'predicted_net_cash']]
                forecast_df_display.columns = ['Month', 'Predicted Net Cash']

            st.table(forecast_df_display)

            forecast_for_plot = forecast_df.rename(columns={'predicted_net_cash': 'net_cash'})[['date', 'net_cash']]
            full = pd.concat([df_month[['date', 'net_cash']], forecast_for_plot], ignore_index=True)
            fig2 = px.line(full, x='date', y='net_cash', title=f"{model_choice} Forecast")
            st.plotly_chart(fig2, use_container_width=True)

            csv = forecast_df_display.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction CSV", csv, file_name='cashflow_forecast.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Forecasting failed: {e}")

# ----------------------------------------------------
# Save Model
# ----------------------------------------------------
if st.button("💾 Save Model"):
    if model is None:
        st.error("No trained model to save.")
    else:
        filename = f"{model_choice.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(model, filename)
        st.success(f"✅ {model_choice} model saved successfully as {filename}!")

# ----------------------------------------------------
# QR Code Sharing Section
# ----------------------------------------------------
with st.expander("📱 Share via QR Code"):
    st.write("Generate a QR code to easily open or share a link (e.g., your dashboard, app, or file).")

    link = st.text_input("🔗 Enter the link to share:", "https://example.com/your-dashboard")

    if st.button("Generate QR Code"):
        qr = qrcode.QRCode(version=1, box_size=8, border=2)
        qr.add_data(link)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), use_column_width=False, width=240)

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.markdown("#### © 2025 Business Cash Flow Forecasting | Powered by Streamlit + Prophet + ARIMA + Random Forest")
