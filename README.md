# Rossmann Store Sales Forecasting

Forecasting daily sales across Rossmann drugstores using temporal calendar seasonality, promotion dynamics, competition proximity, and zero-leakage Out-of-Fold (OOF) store profiling.

---

## Project Overview

The objective is to forecast daily sales across 1,115 Rossmann stores over an unseen 6-week out-of-time test horizon (`2015-06-20` to `2015-07-31`), mirroring the real-world Kaggle competition setup.

We benchmark four approaches:
1. **Regularized Linear Baseline:** Ridge Regression ($\alpha = 10.0$).
2. **Deep Neural Network:** PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization, ReLU activations, and Dropout.
3. **Gradient Boosted Decision Trees:** LightGBM Regressor with early stopping.
4. **Validation-Optimized Ensemble:** Bounded optimization fit on the validation split in log space with Jensen's inequality bias correction.

---

## Key Methodological Principles

1. **Zero-Leakage Out-of-Fold (OOF) Store Profiling:**
   - Individual stores have baseline sales volumes ranging from 3,500 EUR/day to over 25,000 EUR/day.
   - To provide store capacity signals without in-fold target leakage, we compute **Out-of-Fold (OOF) historical store volume statistics** using 5-fold cross-validation on the training period:
     - `Store_Mean_Sales` & `Store_Median_Sales`
     - `Store_Sales_Std`
     - `Store_Mean_Customers` & `Store_Sales_Per_Cust`
     - `Store_Promo_Lift` (store-specific promotion elasticity)
   - These features are log-transformed for additive alignment with the log target ($\log(1 + \text{Sales})$). Full training statistics are mapped to validation and test periods without leakage.

2. **Strict Chronological Validation:**
   - **Training Set:** 2013-01-01 to 2015-05-08.
   - **Validation Set (Dev):** 2015-05-09 to 2015-06-19 (used for early stopping, deep checkpointing, and ensemble weight optimization).
   - **Test Set:** 2015-06-20 to 2015-07-31 (final 6 weeks).

3. **Domain Constraints & Metric Calibration:**
   - **Closed Store Rule:** When `Open == 0`, sales are deterministically 0 EUR. Models train exclusively on open store days (`Open == 1` & `Sales > 0`).
   - **Log-Space Optimization:** Models train on $\log(1 + \text{Sales})$, guaranteeing positive forecasts and stabilizing gradient updates.
   - **Jensen's Inequality Bias Correction:** An asymmetric percentage metric like RMSPE penalizes over-forecasting more heavily than under-forecasting. A scalar calibration factor ($\alpha \approx 0.990$) fit on the validation set corrects for log-normal re-transformation bias.

4. **Rigorous Machine Learning Implementation:**
   - **Deep Checkpointing:** PyTorch model checkpointing uses `copy.deepcopy(model.state_dict())` to prevent in-place weight mutation during subsequent epochs.
   - **Targeted Scaling:** Continuous numerical features are standardized via `StandardScaler` fitted on the training split, leaving binary dummy indicators in their native 0/1 scale.
   - **Clean Competition Status:** `CompetitionStatus` is encoded as clean one-hot categories (`NoCompetition`, `ActiveKnownDate`, `ActiveUnknownDate`, `NotYetOpen`).
   - **True Ensemble Synergy:** Ensemble blend weights ($w \in [0, 1]$) and bias correction are jointly optimized in log space on the validation split:
     $$\hat{y}_{\text{ens}} = \alpha \cdot \left(\exp\left(w \hat{y}_{\text{tree}} + (1 - w) \hat{y}_{\text{nn}}\right) - 1\right)$$

---

## Out-of-Time Test Set Benchmark (Final 6 Weeks)

Evaluated on the unseen test period:

| Model | Test RMSE (EUR) | Test MAE (EUR) | Test RMSPE (%) | Kaggle RMSPE (Ratio) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Ridge Baseline** | 1,255.85 | 806.85 | 20.23% | 0.2023 |
| **2. PyTorch MLP** | 987.87 | 632.38 | 14.87% | 0.1487 |
| **3. LightGBM Decision Tree** | 926.60 | 588.80 | 13.79% | 0.1379 |
| **4. Log-Space Ensemble Blend** | **913.49** | **579.79** | **13.49%** | **0.1349** |

*Note: For comparison, the 1st place competition solution by Gert Jacobusse scored **0.10021**.*

---

## Repository Structure

- [`project.ipynb`](./project.ipynb): Complete end-to-end Jupyter Notebook with full markdown explanations, exploratory data analysis, executed code cells, diagnostics, and plots.
- [`train.csv`](./train.csv): Historical daily sales records per store.
- [`store.csv`](./store.csv): Store metadata (assortment, competition, promotions).
- [`test.csv`](./test.csv): Official test set template.
