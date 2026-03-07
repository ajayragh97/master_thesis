import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Update this path
ESTIMATED_TRAJ_FILE = '/home/ajay/work/temp/aspen_run7/reve/cascade_odometry_full_optimized.txt'
GT_POSES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'

def main():
    est_data = np.loadtxt(ESTIMATED_TRAJ_FILE)
    est_x, est_y, est_z = est_data[:, 1], est_data[:, 2], est_data[:, 3]

    gt_data = np.loadtxt(GT_POSES_FILE)
    gt_x, gt_y, gt_z = gt_data[:, 0], gt_data[:, 1], gt_data[:, 2]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    ax.plot(gt_x, gt_y, gt_z, label='Ground Truth', color='gray', linestyle='--', linewidth=2)
    ax.plot(est_x, est_y, est_z, label='Factor Graph (3D)', color='blue', linewidth=2)

    # Labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Reconstruction')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()