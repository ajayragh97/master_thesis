import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Force an interactive backend if possible ---
# If you are on Linux, 'TkAgg' or 'Qt5Agg' are usually best.
# If this causes an error, comment out the line below.
matplotlib.use('TkAgg') 

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/home/ajay/work/temp/aspen_run7/'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
GT_TUM_FILE = os.path.join(BASE_DIR, 'groundtruth/gt_tum.txt')

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_scenarios_interactive():
    if not os.path.exists(GT_TUM_FILE):
        print(f"Error: GT file not found.")
        return
    gt_data = np.loadtxt(GT_TUM_FILE)
    gt_x, gt_y, gt_z = gt_data[:, 1], gt_data[:, 2], gt_data[:, 3]

    scenarios = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    for scen in scenarios:
        print(f"\nDisplaying Scenario: {scen}")
        print("Close the plot window to save and move to the next scenario.")
        
        scen_path = os.path.join(RESULTS_DIR, scen)
        files = [f for f in os.listdir(scen_path) if f.startswith('skip_') and f.endswith('.txt') and 'temp' not in f]
        
        def sort_key(val):
            rate = val.replace('skip_', '').replace('.txt', '')
            return 999999 if rate == 'blind' else int(rate)
        files.sort(key=sort_key)

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot GT
        ax.plot(gt_x, gt_y, gt_z, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ground Truth')
        ax.scatter(gt_x[0], gt_y[0], gt_z[0], color='green', marker='^', s=100, label='Start')

        color_map = plt.get_cmap('tab10') # Distinct colors
        for i, f in enumerate(files):
            full_path = os.path.join(scen_path, f)
            rate_label = f.replace('.txt', '').replace('skip_', 'Skip: ')
            try:
                data = np.loadtxt(full_path)
                if data.ndim == 1: data = data.reshape(1, -1)
                est_x, est_y, est_z = data[:, 1], data[:, 2], data[:, 3]
                
                # Highlight Blind Mode
                lw = 2.5 if 'blind' in f else 1.2
                ls = '-.' if 'blind' in f else '-'
                
                ax.plot(est_x, est_y, est_z, label=rate_label, linewidth=lw, linestyle=ls)
            except: continue

        ax.set_title(f'3D Comparison: {scen}', fontweight='bold')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        set_axes_equal(ax)
        
        # --- KEY CHANGE: SHOW THEN SAVE ---
        # plt.show() blocks the script until you close the window
        plt.show() 
        
        # Save the current view (if you rotated it, this saves the last view you saw!)
        save_path = os.path.join(RESULTS_DIR, f"{scen}_3d_comparison.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_scenarios_interactive()