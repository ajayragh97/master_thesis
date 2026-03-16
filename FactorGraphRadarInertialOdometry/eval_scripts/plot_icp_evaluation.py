import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Configuration / File Paths
# ==========================================
ICP_TRAJ_FILE = '/home/ajay/work/temp/aspen_run7/reve/icp_raw_trajectory.txt'
GT_POSES_FILE = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'

def main():
    # 1. Load Data
    print("Loading Raw ICP Odometry and Ground Truth...")
    if not os.path.exists(ICP_TRAJ_FILE):
        print(f"Error: Could not find {ICP_TRAJ_FILE}")
        return

    icp_data = np.loadtxt(ICP_TRAJ_FILE)
    t_icp = icp_data[:, 0]
    x_icp, y_icp, z_icp = icp_data[:, 1], icp_data[:, 2], icp_data[:, 3]

    gt_data = np.loadtxt(GT_POSES_FILE)
    x_gt, y_gt, z_gt = gt_data[:, 0], gt_data[:, 1], gt_data[:, 2]

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Raw ICP Odometry vs Ground Truth (No IMU / Factor Graph)', fontsize=16)

    # ---- Plot 1: Top-Down (X-Y) View ----
    ax1.plot(x_gt, y_gt, label='Ground Truth', color='gray', linestyle='--', linewidth=2.5)
    ax1.plot(x_icp, y_icp, label='Raw ICP Odometry', color='red', alpha=0.8, linewidth=2)
    
    # Mark Start Points
    ax1.scatter(x_gt[0], y_gt[0], c='green', marker='^', s=150, zorder=5, label='Start')
    
    ax1.set_title('Top-Down View (X-Y Plane)', fontsize=14)
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.axis('equal') # Keep aspect ratio square!

    # ---- Plot 2: Elevation (Z) over time ----
    # Note: We plot against array index to approximate time progression for GT 
    # since GT might have a different frequency than ICP.
    ax2.plot(np.linspace(0, 100, len(z_gt)), z_gt, label='Ground Truth Z', color='gray', linestyle='--', linewidth=2.5)
    ax2.plot(np.linspace(0, 100, len(z_icp)), z_icp, label='Raw ICP Z', color='red', alpha=0.8, linewidth=2)
    
    ax2.set_title('Elevation Profile (Z-Axis)', fontsize=14)
    ax2.set_xlabel('Trajectory Progression (%)', fontsize=12)
    ax2.set_ylabel('Z (meters)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()