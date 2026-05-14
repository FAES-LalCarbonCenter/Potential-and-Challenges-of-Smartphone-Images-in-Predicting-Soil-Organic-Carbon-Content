
"""
LightGBM — Training and Testing Actual vs Predicted SOC
=========================================================
Journal-compliant figure formatting:
  - Arial (sans-serif) font throughout
  - Consistent 10-11pt font size across all text elements
  - Stats box uses Arial (not monospace)
  - Axis labels not bold
  - No title, no grid, no legend, no 1:1 line
  - Origin at (0, 0), regression line only
"""
 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import pickle
 
# =============================================================================
# GLOBAL FONT SETTINGS — Arial throughout
# =============================================================================
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size']       = 10
mpl.rcParams['axes.titlesize']  = 10
mpl.rcParams['axes.labelsize']  = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
 
# =============================================================================
# PATHS — update if needed
# =============================================================================
PROJECT_ROOT = Path(
    "/Users/dharamkar.1/Library/CloudStorage/"
    "OneDrive-TheOhioStateUniversity/"
    "VSCode_image_processing_project_pipeline"
)
 
MODEL_PATH = (PROJECT_ROOT
              / "notebooks/objective_1/output_data"
              / "step3_output/models/models/LightGBM/model.pkl")
 
TRAIN_PATH = (PROJECT_ROOT
              / "notebooks/objective_1/output_data"
              / "step2_output/calibration_selected_features.csv")
 
TEST_PATH  = (PROJECT_ROOT
              / "notebooks/objective_1/output_data"
              / "step2_output/validation_selected_features.csv")
 
OUT_DIR = (PROJECT_ROOT
           / "notebooks/objective_1/output_data"
           / "output_visualizations/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# =============================================================================
# HELPERS
# =============================================================================
 
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rpd  = y_true.std() / rmse
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse
    reg  = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    return {
        'n':         len(y_true),
        'r2':        r2,
        'rmse':      rmse,
        'mae':       mae,
        'rpd':       rpd,
        'rpiq':      rpiq,
        'slope':     reg.coef_[0],
        'intercept': reg.intercept_,
    }
 
 
def draw_plot(ax, y_true, y_pred):
    """Draw one scatter plot — journal-compliant formatting."""
    m = compute_metrics(y_true, y_pred)
 
    lim_max = max(y_true.max(), y_pred.max()) * 1.05
    x_vals  = np.linspace(0, lim_max, 300)
 
    # Clean white background, no grid
    ax.set_facecolor('white')
    ax.grid(False)
 
    # Scatter
    ax.scatter(y_true, y_pred,
               s=45, alpha=0.65,
               color='#4a90d9',
               edgecolor='none',
               zorder=3)
 
    # Regression line only (no 1:1 line, no legend)
    reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    ax.plot(x_vals, reg.predict(x_vals.reshape(-1, 1)),
            color='#d62728', linewidth=2.0, zorder=4)
 
    # Equal axes from origin
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal', adjustable='box')
 
    # No title; axis labels — Arial, not bold, 10pt (set globally via rcParams)
    ax.set_title('')
    ax.set_xlabel('Actual Soil Organic Carbon (%)')
    ax.set_ylabel('Predicted Soil Organic Carbon (%)')
 
    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
 
    # Stats box — Arial, 10pt, matching all other text
    textstr = (
        f"N = {m['n']}\n"
        f"R\u00b2 = {m['r2']:.2f}\n"
        f"RMSE = {m['rmse']:.2f}\n"
        f"MAE = {m['mae']:.2f}\n"
        f"RPD = {m['rpd']:.2f}\n"
        f"RPIQ = {m['rpiq']:.2f}\n"
        f"y = {m['slope']:.2f}x + {m['intercept']:.2f}"
    )
    ax.text(
        0.03, 0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=10,                # same as all other text
        fontfamily='sans-serif',    # Arial via rcParams
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.92,
            edgecolor='#888888',
            linewidth=1.2
        )
    )
 
 
# =============================================================================
# LOAD MODEL & DATA
# =============================================================================
print("Loading model ...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
 
print("Loading training data ...")
df_train = pd.read_csv(TRAIN_PATH)
y_train  = df_train['soc'].values
X_train  = df_train.drop(columns=['soc'])
yp_train = model.predict(X_train)
 
print("Loading testing data ...")
df_test  = pd.read_csv(TEST_PATH)
y_test   = df_test['soc'].values
X_test   = df_test.drop(columns=['soc'])
yp_test  = model.predict(X_test)
 
print(f"Training: n={len(y_train)}  |  Testing: n={len(y_test)}")
 
# =============================================================================
# PLOT 1 — Side-by-side (Training + Testing)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('white')
 
draw_plot(axes[0], y_train, yp_train)
draw_plot(axes[1], y_test,  yp_test)
 
plt.tight_layout(w_pad=4)
combined_path = OUT_DIR / "LightGBM_Train_Test_Combined.png"
plt.savefig(combined_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {combined_path}")
 
# =============================================================================
# PLOT 2 — Individual: Training
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('white')
draw_plot(ax, y_train, yp_train)
plt.tight_layout()
train_path = OUT_DIR / "LightGBM_Training.png"
plt.savefig(train_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {train_path}")
 
# =============================================================================
# PLOT 3 — Individual: Testing
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('white')
draw_plot(ax, y_test, yp_test)
plt.tight_layout()
test_path = OUT_DIR / "LightGBM_Testing.png"
plt.savefig(test_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {test_path}")
 
print("\nDone. Three files generated:")
print(f"  {combined_path}")
print(f"  {train_path}")
print(f"  {test_path}")
 