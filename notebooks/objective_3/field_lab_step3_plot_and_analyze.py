"""
SOIL TYPE ANALYSIS - STEP 3: Plot and Analyze
==============================================
# """
"""
Objective 3 — Field vs Lab Image SOC Plots
===========================================
Journal-compliant figure formatting:
  - Arial (sans-serif) font throughout
  - Consistent 10pt font size across all text elements
  - Stats box uses Arial (not monospace)
  - Axis labels not bold
  - No title, no grid, no legend, no 1:1 line
  - Origin at (0, 0), regression line only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

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
# PATHS
# =============================================================================
PROJECT_ROOT = Path(
    "/Users/dharamkar.1/Library/CloudStorage/"
    "OneDrive-TheOhioStateUniversity/"
    "VSCode_image_processing_project_pipeline"
)

PRED_CSV = (PROJECT_ROOT
            / "notebooks/objective_3/obj3_output_data"
            / "step2_output/lightgbm_field_lab_predictions.csv")

OUT_DIR  = (PROJECT_ROOT
            / "notebooks/objective_3/obj3_output_data"
            / "step3_output/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rpd  = np.std(y_true, ddof=1) / rmse if rmse > 0 else np.nan
    q75, q25 = np.percentile(y_true, [75, 25])
    rpiq = (q75 - q25) / rmse if rmse > 0 else np.nan
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
    """Shared consistent plot style — journal compliant."""
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

    # Regression line only
    reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred)
    ax.plot(x_vals, reg.predict(x_vals.reshape(-1, 1)),
            color='#d62728', linewidth=2.0, zorder=4)

    # Equal axes from origin
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal', adjustable='box')

    # No title; axis labels — Arial, not bold (picked up from rcParams)
    ax.set_title('', pad=20)
    ax.set_xlabel('Actual Soil Organic Carbon (%)')
    ax.set_ylabel('Predicted Soil Organic Carbon (%)')

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Stats box — Arial 10pt, matching all other text
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
        fontsize=10,
        fontfamily='sans-serif',
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


def make_plot(df_subset, filename):
    """Create and save one plot."""
    if df_subset.empty:
        print(f"  Skipping {filename} — no samples")
        return

    y_true = df_subset['soc_actual'].values
    y_pred = df_subset['soc_predicted'].values

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(top=0.92)

    draw_plot(ax, y_true, y_pred)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# LOAD & PLOT
# =============================================================================
print("Loading predictions...")
df = pd.read_csv(PRED_CSV)
print(f"  Total rows: {len(df)}")
print(f"  Columns:    {df.columns.tolist()}")

print("\nCreating plots...")
make_plot(df[df['image_type'] == 'Field'], 'soc_field.png')
make_plot(df[df['image_type'] == 'Lab'],   'soc_lab.png')

print("\nDone.")