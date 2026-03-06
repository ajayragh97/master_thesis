import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Configuration / File Paths
# ==========================================
# Update these paths to point to your actual files
ESTIMATED_TRAJ_FILE = '/home/ajay/work/temp/aspen_run7/reve/aspen_run7_single_chip_odometry_real_time_optimized.txt'
GT_POSES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'
GT_TIMES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/timestamps.txt'

def main():
    # 1. Check if files exist
    for f in[ESTIMATED_TRAJ_FILE, GT_POSES_FILE, GT_TIMES_FILE]:
        if not os.path.exists(f):
            print(f"Error: File '{f}' not found!")
            return

    # 2. Load the data
    print("Loading data...")
    # Estimated: timestamp, x, y, z, qx, qy, qz, qw
    est_data = np.loadtxt(ESTIMATED_TRAJ_FILE)
    est_t = est_data[:, 0]
    est_x = est_data[:, 1]
    est_y = est_data[:, 2]

    # Groundtruth Poses: x, y, z, qx, qy, qz, qw
    gt_poses = np.loadtxt(GT_POSES_FILE)
    gt_x = gt_poses[:, 0]
    gt_y = gt_poses[:, 1]

    # Groundtruth Timestamps: timestamp
    gt_t = np.loadtxt(GT_TIMES_FILE)

    # Make sure GT poses and timestamps match in length
    if len(gt_t) != len(gt_poses):
        print(f"Warning: GT timestamps ({len(gt_t)}) and GT poses ({len(gt_poses)}) have different lengths!")

    # 3. Calculate End Point Error (2D)
    # Find the closest ground truth timestamp to the last estimated timestamp
    last_est_t = est_t[-1]
    idx_gt = np.argmin(np.abs(gt_t - last_est_t))
    
    err_x = est_x[-1] - gt_x[idx_gt]
    err_y = est_y[-1] - gt_y[idx_gt]
    epe_2d = np.sqrt(err_x**2 + err_y**2)
    duration = est_t[-1] - est_t[0]

    print(f"Trajectory Duration: {duration:.2f} seconds")
    print(f"End Point Error (2D): {epe_2d:.2f} meters")

    # 4. Plotting
    plt.figure(figsize=(12, 8))

    # Plot trajectories
    plt.plot(gt_x, gt_y, color='dimgray', linestyle='--', linewidth=2.5, label='Ground Truth')
    plt.plot(est_x, est_y, color='blue', linestyle='-', linewidth=2, label='Factor Graph Estimate')

    # Plot Start and End markers
    plt.scatter(gt_x[0], gt_y[0], c='green', marker='^', s=150, zorder=5, label='GT Start')
    plt.scatter(gt_x[-1], gt_y[-1], c='green', marker='x', s=150, zorder=5, label='GT End')
    
    plt.scatter(est_x[0], est_y[0], c='red', marker='^', s=150, zorder=5, label='Est Start')
    plt.scatter(est_x[-1], est_y[-1], c='red', marker='x', s=150, zorder=5, label='Est End')

    # Add Error Text Box
    textstr = f'End Point Error (2D): {epe_2d:.2f} m'
    props = dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    # Formatting
    plt.title(f'2D Trajectory Analysis\nDuration: {duration:.2f} seconds', fontsize=14)
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    
    # Legend and Grid
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Force equal aspect ratio so turns aren't stretched/distorted!
    plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()