import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set page layout
st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("Sales Forecasting with XGBoost")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    lag = st.slider("Number of lag days", min_value=1, max_value=15, value=5)
    forecast_days = st.slider("Days to forecast", min_value=1, max_value=30, value=7)
    show_raw = st.checkbox("Show raw dataset", value=False)

# Load and preprocess data
file_path = 'train.csv'
data = pd.read_csv(file_path)
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce', dayfirst=True)
data = data.dropna(subset=['Order Date'])
sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()

# Show raw data
if show_raw:
    st.subheader("Raw Dataset")
    st.dataframe(sales_by_date.head(20), use_container_width=True)

# Plot sales trend
st.subheader("Sales Trend Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(sales_by_date['Order Date'], sales_by_date['Sales'], color='darkred')
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
ax1.set_title("Historical Sales")
ax1.grid(True)
st.pyplot(fig1)

# Feature engineering
def create_lagged_features(data, lag=1):
    lagged = data.copy()
    for i in range(1, lag + 1):
        lagged[f'lag_{i}'] = lagged['Sales'].shift(i)
    return lagged

sales_with_lags = create_lagged_features(sales_by_date, lag).dropna().reset_index(drop=True)
X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
y = sales_with_lags['Sales']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

st.subheader("Model Performance")
st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

# Plot actual vs predicted
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(y_test.values, label='Actual', color='black')
ax2.plot(predictions, label='Predicted', color='green')
ax2.set_title("Model Prediction on Test Set")
ax2.set_xlabel("Index")
ax2.set_ylabel("Sales")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Forecast future sales
latest_lags = sales_with_lags.iloc[-1][[f'lag_{i}' for i in range(1, lag + 1)]].values.tolist()
future_sales = []

for _ in range(forecast_days):
    input_array = np.array(latest_lags[-lag:]).reshape(1, -1)
    pred = model.predict(input_array)[0]
    future_sales.append(pred)
    latest_lags.append(pred)

last_known_date = sales_by_date['Order Date'].max()
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_days)

forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Sales': future_sales})

st.subheader("Forecast Future Sales by Year")

selected_year = st.number_input("Enter a future year to forecast (e.g. 2025):", min_value=2024, max_value=2100, value=2025)

if selected_year <= sales_by_date['Order Date'].dt.year.max():
    st.warning("Please enter a year *after* the latest date in the dataset.")
else:
    # Days to forecast for selected year
    days_to_forecast = 366 if pd.Timestamp(f"{selected_year}-12-31").is_leap_year else 365

    # Forecasting loop
    recent_lags = sales_with_lags.iloc[-1][[f'lag_{i}' for i in range(1, lag + 1)]].values.tolist()
    forecast_values = []

    for _ in range(days_to_forecast):
        input_array = np.array(recent_lags[-lag:]).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        forecast_values.append(prediction)
        recent_lags.append(prediction)

    # Generate forecast dates
    last_date = sales_by_date['Order Date'].max()
    start_date = pd.Timestamp(f"{selected_year}-01-01")
    forecast_dates = pd.date_range(start=start_date, periods=days_to_forecast)

    forecast_year_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Sales': forecast_values
    })

    # Plotting
    st.subheader(f"Sales Forecast for {selected_year}")
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(forecast_year_df['Date'], forecast_year_df['Forecasted Sales'], color='blue')
    ax4.set_title(f"Forecasted Sales in {selected_year}")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Predicted Sales")
    ax4.grid(True)
    st.pyplot(fig4)

    # Table view
    with st.expander("Show Forecast Table"):
        st.dataframe(forecast_year_df)
