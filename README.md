# Sentimen Analysis on Keluhan Istri Data

- **Dataset:** [Airbnb NYC 2019](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
- **Tasks:** Indonesian Text Cleaning, Sentiment Modeling
- **Metrics:** F1 Score
- **Goal:**

## 📌 Overview

- **Problem:** Increasing RMSE score for AirBnB price prediction, by doing some Feature Engineering
- **Dataset:** [New York City AirBnB Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **Approach:** EDA → Feature Engineering → Modeling → Evaluation → Deployment.
- **Tools:** Python, Pandas, Scikit-learn.

## 📊 Exploratory Data Analysis

Key insights from data exploration:

- Distribution of target/labels.
- Missing values or outliers.
- Interesting correlations or trends.

📷 _[Insert example plots or figures here]_

## 🤖 Models & Techniques

- **Baseline Model:** (e.g., Logistic Regression, ARIMA) → metric score.
- **Advanced Model(s):** (e.g., XGBoost, LSTM, BERT) → metric score.
- **Feature Engineering:** Describe any created/engineered features.
- **Hyperparameter Tuning:** Grid search / Random search / Bayesian optimization.

📊 _Comparison Table:_

| Model            | Metric (e.g., RMSE / F1) |
| ---------------- | ------------------------ |
| Baseline         | 0.512                    |
| Advanced Model 1 | 0.431                    |
| Advanced Model 2 | 0.387 ✅                 |

## 📈 Results

- Main performance results.
- Visualizations (e.g., confusion matrix, SHAP plots, prediction samples).

📷 _[Insert results screenshot or chart]_

## 🔍 Explainability

- Feature importance / SHAP summary.
- Example of model interpretability.

📷 _[Insert SHAP plot / attention heatmap]_

## 🌐 Deployment

- **Demo App:** [Link to Streamlit/Gradio app](#)
- **Run locally:**
  ```bash
  pip install -r requirements.txt
  streamlit run app/app.py
  ```
