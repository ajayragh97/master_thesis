import os
import numpy as np
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = '/media/ajay/Backup_Plus1/datasets/coloradar/kitti/2_28_2021_outdoors_run7/'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
GT_POSES = os.path.join(BASE_DIR, 'groundtruth/groundtruth_poses.txt')
GT_TIMES = os.path.join(BASE_DIR, 'groundtruth/timestamps.txt')
GT_TUM = os.path.join(BASE_DIR, 'groundtruth/gt_tum.txt')

# The distance over which to calculate drift (Standard Odometry Evaluation)
DELTA_DIST = 10 

def prepare_gt():
    """Merges timestamps and poses into one TUM file for evo."""
    if os.path.exists(GT_TUM): return
    print("Preparing Ground Truth TUM file...")
    times = np.loadtxt(GT_TIMES); poses = np.loadtxt(GT_POSES)
    np.savetxt(GT_TUM, np.column_stack((times, poses)), fmt='%.6f')

def get_evo_metric(gt_file, est_file, mode='trans'):
    """Calls evo_rpe and parses the RMSE from stdout."""
    r_flag = 'trans_part' if mode == 'trans' else 'angle_deg'
    cmd = ["evo_rpe", "tum", gt_file, est_file, "-r", r_flag, "--delta", str(DELTA_DIST), "--delta_unit", "m"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0: return None
        match = re.search(r"rmse\s+([\d\.]+)", result.stdout, re.IGNORECASE)
        return float(match.group(1)) if match else None
    except: return None

def plot_drift_results(df, output_dir):
    """Generates visual plots for Translation and Rotation drift."""
    # Filter out failures
    success_df = df.dropna(subset=['Drift (%)']).copy()
    
    # Separate numeric skip rates from "blind" for the line plot
    numeric_df = success_df[success_df["Skip_Rate"] != "blind"].copy()
    numeric_df["Skip_Rate"] = pd.to_numeric(numeric_df["Skip_Rate"])
    
    scenarios = success_df["Scenario"].unique()
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd'] # Blue, Green, Red, Purple
    markers = ['o', 's', '^', 'D']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Odometry Drift Analysis (Evaluated over {DELTA_DIST}m segments)', fontsize=16, fontweight='bold')

    for i, scen in enumerate(scenarios):
        scen_data = numeric_df[numeric_df["Scenario"] == scen].sort_values("Skip_Rate")
        blind_data = success_df[(success_df["Scenario"] == scen) & (success_df["Skip_Rate"] == "blind")]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # --- Subplot 1: Translation Drift % ---
        ax1.plot(scen_data["Skip_Rate"], scen_data["Drift (%)"], marker=marker, color=color, label=scen, linewidth=2)
        if not blind_data.empty:
            ax1.scatter([2000], blind_data["Drift (%)"], color=color, marker='*', s=200, edgecolors='black', zorder=5)

        # --- Subplot 2: Rotation Error (deg/m) ---
        ax2.plot(scen_data["Skip_Rate"], scen_data["Rot Error (deg/m)"], marker=marker, color=color, label=scen, linewidth=2)
        if not blind_data.empty:
            ax2.scatter([2000], blind_data["Rot Error (deg/m)"], color=color, marker='*', s=200, edgecolors='black', zorder=5)

    # Formatting Ax1 (Translation)
    ax1.set_xscale('log'); ax1.set_title('Translation Drift (%)'); ax1.set_ylabel('Drift Percentage (%)')
    ax1.set_xlabel('GT Skip Rate (Log Scale)'); ax1.grid(True, which="both", ls="-", alpha=0.3); ax1.legend()

    # Formatting Ax2 (Rotation)
    ax2.set_xscale('log'); ax2.set_title('Rotation Drift (deg/m)'); ax2.set_ylabel('Degrees per Meter')
    ax2.set_xlabel('GT Skip Rate (Log Scale)'); ax2.grid(True, which="both", ls="-", alpha=0.3); ax2.legend()

    # Add Footnote for blind mode
    plt.figtext(0.5, 0.02, 'Note: Stars (*) at the far right represent "Blind" mode (No GT after initialization)', 
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plot_path = os.path.join(output_dir, "odometry_drift_visual.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Visual plot saved to: {plot_path}")
    plt.show()

def main():
    prepare_gt()
    summary_data = []
    
    if not os.path.exists(RESULTS_DIR): return
    scenarios = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    for scen in scenarios:
        scen_path = os.path.join(RESULTS_DIR, scen)
        traj_files = [f for f in os.listdir(scen_path) if f.startswith('skip_') and f.endswith('.txt') and 'temp_evo' not in f]
        print(f"\nEvaluating: {scen}")

        for f in traj_files:
            skip_rate_str = f.replace('skip_', '').replace('.txt', '')
            full_path = os.path.join(scen_path, f); temp_path = os.path.join(scen_path, f"temp_evo_{scen}_{skip_rate_str}.txt")
            try:
                data = np.loadtxt(full_path); np.savetxt(temp_path, data[:, :8] if data.ndim > 1 else data.reshape(1,-1)[:, :8], fmt='%.6f')
                trans_rmse = get_evo_metric(GT_TUM, temp_path, mode='trans')
                rot_rmse = get_evo_metric(GT_TUM, temp_path, mode='rot')
                if trans_rmse is not None:
                    drift_pct = (trans_rmse / DELTA_DIST) * 100; deg_per_m = (rot_rmse / DELTA_DIST)
                    print(f"  -> {f}: Drift {drift_pct:.3f}%")
                    summary_data.append({"Scenario": scen, "Skip_Rate": skip_rate_str, "Drift (%)": drift_pct, "Rot Error (deg/m)": deg_per_m})
                if os.path.exists(temp_path): os.remove(temp_path)
            except Exception as e: print(f"  -> ERROR {f}: {e}")

    if not summary_data: return
    df = pd.DataFrame(summary_data)
    df['sort'] = df['Skip_Rate'].apply(lambda x: 999999 if x == 'blind' else int(x))
    df = df.sort_values(by=['Scenario', 'sort']).drop(columns=['sort'])
    
    print("\n" + "="*90 + "\nDRIFT LEADERBOARD\n" + "="*90)
    print(df.to_string(index=False, na_rep='N/A'))
    
    df.to_csv(os.path.join(RESULTS_DIR, "final_drift_report.csv"), index=False)
    plot_drift_results(df, RESULTS_DIR)

if __name__ == "__main__":
    main()