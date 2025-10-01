import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats 

# Initialize an empty list to hold all pixel differences
all_values = []

tif_files = glob.glob("comparison/compare_input_1/*.tif")

# -----------------
# 1. DATA AGGREGATION LOOP
# -----------------
for i in tif_files:
    # 1. Open files and get data/metadata
    base_name = os.path.splitext(os.path.basename(i))[0]
    soft_path = os.path.join("comparison/compare_input_2", f"{base_name}.tiff")

    try:
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
            
        # 4. Filter data
        values = diff[mask & ~np.isnan(diff)]

        # Append the filtered values from this file to the master list
        if len(values) > 0:
            all_values.append(values)
        else:
            print(f"Warning: No valid data found for {base_name}.")

    except rasterio.RasterioIOError:
        print(f"Error: Could not open one or both files for {base_name}. Skipping.")
        
# -----------------
# 2. OVERALL PLOTTING AND STATISTICS
# -----------------
# Combine all arrays in the list into a single NumPy array
if not all_values:
    print("Error: No valid pixel data was collected from any file.")
else:
    overall_values = np.concatenate(all_values)

    # --- STATISTICS CALCULATION ---
    mean_val = np.mean(overall_values)
    mode_val = stats.mode(overall_values, keepdims=False)[0]
    variance_val = np.var(overall_values)
    sd_val = np.std(overall_values)

    stats_text = (
        f"Mean: {mean_val:.3f}\n"
        f"Mode: {mode_val:.3f}\n"
        f"Variance: {variance_val:.3f}\n"
        f"SD: {sd_val:.3f}"
    )

    # 5. Plot and save
    plt.figure(figsize=(10,6))
    sns.histplot(overall_values, bins=50, kde=True, color="darkgreen", stat="density")
    
    plt.title("Overall Distribution of Pixel Differences Across All Files")
    plt.xlabel("Pixel Difference Value")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Add text box to the top right corner
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )

    plt.savefig("comparison/Overall_Diff_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close()