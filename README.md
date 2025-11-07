# Medical Insurance Cost Prediction

## Overview
This project develops an interpretable machine learning pipeline to predict individual medical insurance charges and identify key lifestyle and demographic factors influencing healthcare costs.  
The main objectives are to maintain **accuracy**, **interpretability**, and **fairness** across subgroups while keeping the model transparent.

---

## Dataset Information

- **Rows:** 1,338  
- **Columns:** 7  
- **Target Variable:** `charges` (individual medical insurance cost)

### Features
- `age`  
- `sex`  
- `BMI`  
- `children`  
- `smoker` status  
- `region`  

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Explored feature distributions, correlations, and subgroup patterns  
- Identified `smoker` status and `BMI` as dominant predictors of insurance cost  

### 2. Model Development
- **Data Preprocessing & Feature Engineering**
  - Standardized continuous features (`age`, `BMI`)  
  - Encoded categorical variables (`smoker`, `region`)  
  - Added interaction terms to capture non-linear effects:
    - `smoker × BMI`  
    - `smoker × age`  

- **Model Training**
  - Applied and compared multiple regression and ensemble approaches using **cross-validation** and **grid search**

| Rank | Model Type        | Algorithm / Variant                                                        | MAE (train ± std) | MAE (test ± std) | R² (test ± std) | Min Residual |
|------|-----------------|---------------------------------------------------------------------------|-----------------|----------------|----------------|--------------|
| 🥇   | Tree-Based Ensemble | CatBoost                                                                  | 1566.37 ± 52.28 | 1405.26 ± 228.92 | 0.870 ± 0.028 | -161.11 |
| 🥈   | Tree-Based Ensemble | LightGBM                                                                  | 1647.14 ± 60.28 | 1471.65 ± 234.88 | 0.870 ± 0.027 | -175.49 |
| 🥉   | Tree-Based Ensemble | XGBoost                                                                   | 1745.51 ± 65.78 | 1548.14 ± 203.61 | 0.868 ± 0.026 | -197.37 |
| 4    | Tree-Based Ensemble | Extra Trees                                                               | 1787.69 ± 45.76 | 1611.72 ± 216.90 | 0.870 ± 0.027 | -175.97 |
| 5    | Tree-Based Ensemble | Random Forest                                                             | 1823.08 ± 60.43 | 1646.11 ± 167.16 | 0.870 ± 0.025 | -176.97 |
| 6    | Rule-Based Model    | Rule-Based                                                                | 2505.36 ± 81.35 | 2394.46 ± 141.93 | 0.878 ± 0.021 | -110.89 |
| 7    | Linear Model        | Linear Regression                                                         | 2957.03 ± 99.92 | 2791.52 ± 202.74 | 0.853 ± 0.034 | -165.51 |
| 8    | Linear Model        | Lasso                                                                     | 2954.06 ± 100.03| 2793.24 ± 199.83 | 0.853 ± 0.034 | -160.82 |
| 9    | Linear Model        | Ridge                                                                     | 2951.80 ± 98.44 | 2793.41 ± 200.03 | 0.853 ± 0.033 | -158.38 |
| 10   | Linear Model        | ElasticNet                                                                | 2951.82 ± 98.46 | 2794.73 ± 200.20 | 0.853 ± 0.033 | -157.08 |
| 11   | Linear Model        | Polynomial Regression                                                     | 3008.81 ± 84.42 | 2850.33 ± 179.56 | 0.835 ± 0.027 | -158.49 |

---

## Results
- **Best R²:** > 0.85  
- **Mean Absolute Error (MAE):** ~1400  
- **Residuals:** Stable and normally distributed  
- **Top Predictors:** `smoker`, `BMI`, `age`  

---

## Key Insights
- Smoking significantly increases predicted insurance charges  
- BMI interacts with smoking, amplifying health-related cost risks  
- Ensemble methods such as **XGBoost** provide the best trade-off between bias and variance, while **CatBoost** gave the lowest MAE for test data

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries:**
  - Data manipulation & analysis: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Regression & feature engineering: `scikit-learn`  
  - Tree-based ensembles: `xgboost`, `catboost`, `lightgbm`  
  - Model evaluation: `scikit-learn` metrics
 
---

## Future Improvements
- Integrate **SHAP**, **LIME** for better model interpretability  
- Evaluate fairness metrics across demographic subgroups (e.g., gender, region)  
- Deploy the model as an interactive web application (e.g., **Streamlit** or **Flask**)  
