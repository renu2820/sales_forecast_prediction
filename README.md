# Sales Forecasting App

A clean and interactive web application that predicts future sales trends using historical data and machine learning (XGBoost). Built using Streamlit.

---

## 🚀 Live Demo

👉 [Click here to open the app](https://renu2820-sales-forecast-prediction.streamlit.app)

---

<details>
<summary><strong>📌 Features</strong></summary>

- Streamlit-based interactive user interface
- Sales forecasting with XGBoost Regressor
- Trend visualization using time series plots
- Simple input pipeline using historical CSV file
- Easy deployment with Streamlit Cloud

</details>

---

<details>
<summary><strong>📊 Example Chart</strong></summary>

![Forecast Chart](https://github.com/renu2820/sales_forecast_prediction/assets/your_chart_placeholder.png)

> Replace the above image with an actual chart screenshot after deployment.

</details>

---

## 🧠 Tech Stack

| Tool        | Purpose                  |
|-------------|---------------------------|
| Python      | Core programming language |
| Pandas      | Data manipulation         |
| XGBoost     | Forecasting model         |
| Matplotlib  | Visualization             |
| Streamlit   | Web app framework         |

---

## 📂 Dataset Requirements

Ensure your dataset meets the following:

- File name: `train.csv`
- Columns:
  - `Order Date` (format: `DD/MM/YYYY`)
  - `Sales` (numeric)

---

## 🛠 How to Run Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/renu2820/sales_forecast_prediction.git
cd sales_forecast_prediction

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
