# SOC Prediction from Soil Images - Optimized Pipeline

Complete, reproducible pipeline for predicting Soil Organic Carbon (SOC) from soil image features using Multi layer feature selection and multiple ML models selected by AutoML.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- `soil_image_features_without_commas.csv`
- `image_with_soc_metadata.csv`

  Best Model according to AutoML based on only RMSE : WeightedEnsemble_L2
  R²   = 0.7957
  RMSE = 0.5021
  MAE  = 0.1855
  RPD  = 2.2123
  RPIQ = 1.8322

 

  ENSEMBLE COMPOSITION
================================================================================
  WeightedEnsemble_L2 combines:
    - NeuralNetTorch:  91.7%
    - LightGBMLarge:    8.3%


    TOP 10 MODELS:
                 model  score_val  pred_time_val
0  WeightedEnsemble_L2  -0.462987       0.014328
1       NeuralNetTorch  -0.464283       0.009558
2             CatBoost  -0.582114       0.003271
3        LightGBMLarge  -0.592685       0.004190
4              XGBoost  -0.676229       0.002924
5        ExtraTreesMSE  -0.606122       0.027297
6             LightGBM  -0.685139       0.000586
7      RandomForestMSE  -0.624226       0.028064
8           LightGBMXT  -0.642017       0.000810



MODEL PERFORMANCE SUMMARY
================================================================================

              Model       R²     RMSE      MAE      RPD     RPIQ
           LightGBM 0.817497 0.474560 0.289961 2.340806 1.938637
            XGBoost 0.810101 0.484082 0.192766 2.294764 1.900506
           CatBoost 0.805358 0.490089 0.213062 2.266635 1.877209
WeightedEnsemble_L2 0.795674 0.502133 0.185503 2.212268 1.832183
     NeuralNetTorch 0.778849 0.522397 0.188784 2.126453 1.761111
      ExtraTreesMSE 0.766348 0.536960 0.259437 2.068784 1.713350
      LightGBMLarge 0.744586 0.561408 0.245047 1.978691 1.638736
    RandomForestMSE 0.731366 0.575755 0.268214 1.929387 1.597903
         LightGBMXT 0.723613 0.584003 0.352609 1.902135 1.575333

================================================================================


weights = {
    'RPD':   0.35,   # Soil science gold standard
    'RPIQ':  0.25,   # Robust performance indicator
    'RMSE':  0.20,   # Prediction error magnitude
    'R²':    0.15,   # Variance explained
    'MAE':   0.05    # Least critical
}
