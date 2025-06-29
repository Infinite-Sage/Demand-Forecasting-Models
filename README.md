# Explaining Demand-Forecasting-Models for E-commerce (Home Decor Category)
This project uses Explainable AI (XAI) to analyze customer demand for the `home_decor` category using two ML models — Random Forest and KNN — and SHAP values for interpretability.

## Project Objectives

- Identify key drivers behind customer purchases of `home_decor`
- Compare feature importance across Random Forest and KNN models
- Communicate model insights clearly to technical and non-technical teams
- Help marketing, inventory, and strategy teams make smarter decisions

## Files Included

| File | Description |
|------|-------------|
| `model.pkl` | Trained RandomForestRegressor |
| `knn_model.pkl` | Trained KNeighborsRegressor |
| `X_train.csv` | Training data |
| `X_test.csv` | Test data |
| `explaining_demand_forecasting.ipynb` | Final project notebook |

## Methodology

- **XAI Technique Used:** `SHAP` (Shapley values from game theory)
- **Model Comparison:** Cosine similarity used to compare top 5 features
- **Result Consistency:** 0.96 → Stable & Reliable

## Top Predictive Features

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `lag2` | Sales 2 months ago |
| 2 | `sma_4m` | 4-month moving average |
| 3 | `lag1` | Sales 1 month ago |
| 4 | `sma_2m` | 2-month moving average |
| 5 | `sma_2m__colourful_essentials` | Recent category spend |

## 📈 Business Impact

- **Marketing** can run campaigns on customers with strong past purchases
- **Inventory** teams can stock based on predicted category interest
- **Strategy** teams can build bundling offers (e.g. decor + accessories)

## Consistency Score Between Models: `0.96`
✔ Interpretation is reliable across both models

## Tech Stack

- Python 3.10
- SHAP
- scikit-learn
- NumPy, Pandas
- Matplotlib (optional SHAP plot)
