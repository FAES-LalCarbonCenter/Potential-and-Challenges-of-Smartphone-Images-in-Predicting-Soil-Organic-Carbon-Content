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


The code and the respective output for each objective is in its respective folder as Objective 1,Objective 2 and Objective 3


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
