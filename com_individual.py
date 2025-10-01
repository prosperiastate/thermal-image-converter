import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats # <-- New import for mode calculation

tif_files = glob.glob("comparison/compare_input_1/*.tif")

for i in tif_files:
    # 1. Open files and get data/metadata
    base_name = os.path.splitext(os.path.basename(i))[0]
    soft_path = os.path.join("comparison/compare_input_2", f"{base_name}.tiff")

    with rasterio.open(i) as src1, rasterio.open(soft_path) as src2:
        img1, nodata1 = src1.read(1), src1.nodata
        img2, nodata2 = src2.read(1), src2.nodata

    # 2. Calculate difference and initialize mask
    diff = (img1.astype(np.float32) - img2.astype(np.float32)).flatten()
    mask = np.ones(diff.shape, dtype=bool)

    # 3. Apply nodata masks
    if nodata1 is not None:
        mask &= (img1.flatten() != nodata1)
    if nodata2 is not None:
        mask &= (img2.flatten() != nodata2)
        
    # 4. Filter data (apply mask and remove NaN in one step)
    values = diff[mask & ~np.isnan(diff)]
    
    # --- STATISTICS CALCULATION ---
    # Check if there are values before calculating statistics
    if len(values) == 0:
        print(f"Warning: No valid data for {base_name}. Skipping.")
        continue
        
    mean_val = np.mean(values)
    mode_val = stats.mode(values, keepdims=False)[0] # Calculate the mode
    variance_val = np.var(values)
    sd_val = np.std(values)

    # Format the text box for the plot
    stats_text = (
        f"Mean: {mean_val:.3f}\n"
        f"Mode: {mode_val:.3f}\n"
        f"Variance: {variance_val:.3f}\n"
        f"SD: {sd_val:.3f}"
    )

    # 5. Plot and save
    plt.figure(figsize=(8,5))
    sns.histplot(values, bins=50, kde=True, color="purple", stat="density")
    plt.title(f"Distribution of Pixel Differences of {os.path.basename(i)}")
    plt.xlabel("Pixel Difference Value")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # --- ADD STATISTICS TO PLOT ---
    # Add text box to the top right corner of the plot
    # The 'transform=plt.gca().transAxes' makes the coordinates relative (0 to 1)
    plt.text(
        0.98,          # X position (98% of the way across)
        0.98,          # Y position (98% of the way up)
        stats_text,    # The text to display
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7) # Light background box
    )

    plt.savefig(f"comparison/diff_distribution_{base_name}.png", dpi=300, bbox_inches="tight")
    plt.close()