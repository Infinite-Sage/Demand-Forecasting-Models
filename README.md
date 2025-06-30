# Explainable Demand Forecasting for E-Commerce Using SHAP (Home Decor)

## TL;DR

This project applies Explainable AI (XAI) to uncover the key drivers behind customer purchases in the home_decor category. Using two machine learning models — Random Forest and K-Nearest Neighbors (KNN) — we apply SHAP (Shapley values) to make model predictions transparent and actionable for marketing, inventory, and strategy teams.

## Project Objectives

- Identify top features influencing next-month home decor purchases
- Compare feature importance across Random Forest and KNN models
- Translate model predictions into business-friendly insights
- Enable marketing, inventory, and strategy teams to make data-driven decisions

## Project Structure

| File | Description |
|------|-------------|
| `model.pkl` | Trained RandomForestRegressor |
| `knn_model.pkl` | Trained KNeighborsRegressor |
| `X_train.csv` | Features used to train models |
| `X_test.csv` | Features used for explanation |
| `y_train.csv` | Labels (home decor sales) – used for model training |
| `y_test.csv` | Labels for evaluation (not needed for SHAP) |
| `shap_summary_plot.png` | SHAP summary plot |
| `explaining_demand_forecasting.ipynb` | Notebook with SHAP implementation and insights |

## Methodology

- **Explainability Technique:** SHAP — a model-agnostic approach based on cooperative game theory
Models Used:

- RandomForestRegressor — tree-based model explained with TreeExplainer
- KNeighborsRegressor — non-parametric model explained with KernelExplainer

- **Comparison Metric:** Cosine Similarity of top 5 features → Validates consistency across models
- **Reliability Threshold:** Consistency ≥ 0.80 is considered stable
### Result: 0.96 consistency score → interpretations are highly reliable

## Top Predictive Features

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `lag2` | Home decor purchase 2 months ago |
| 2 | `sma_4m` | 4-month moving average of purchases |
| 3 | `lag1` | Home decor purchase 1 month ago |
| 4 | `sma_2m` | 2-month moving average |
| 5 | `sma_2m__colourful_essentials` | Recent spend in related category (cross-sell signal) |

### Business Insights

- **Marketing**: Target users who recently bought decor or similar items
- **Inventory**: Stock planning guided by moving averages and lag trends
- **Strategy**: Cross-sell through category bundles (e.g., Home Decor + Colourful Essentials)

These insights are data-backed and derived from real customer behavior.

## Visualizations
The SHAP summary plot clearly shows which features increase or decrease purchase predictions. Red points = higher feature values (e.g., high spending), blue = low values.

### SHAP Summary Plot

This plot shows which features most impact the Random Forest model’s predictions.

![SHAP Summary](shap_summary_plot.png)

## Tech Stack

- Python 3.10
- SHAP
- scikit-learn
- NumPy, Pandas
- Matplotlib (optional SHAP plot)

### Future Work

- Try time-series specific models (e.g., XGBoost + lag features)
- Scale SHAP explanations on full dataset
- Add LIME as an alternate explainer for model comparison
- Integrate insights into CRM dashboard for real-time decision-making

## Notes
- The y_train.csv and y_test.csv files are only needed if retraining the model. SHAP doesn't require y values directly.
- Data is sampled for speed — feel free to scale up using full sets.
- For public sharing, redact or mask any sensitive or proprietary data.
