import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import os
import argparse

def wrap_angle_pi(angle):
    """ Wraps an angle in radians to the [-pi, pi] range """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_rmse(errors):
    """ Calculates Root Mean Square Error """
    return np.sqrt(np.mean(np.square(errors)))

def evaluate_trajectory(est_file, gt_pose_file, gt_time_file, show_plots=True):
    # 1. Load Data
    if not os.path.exists(est_file):
        print(f"Error: Could not find {est_file}")
        return None

    est_data = np.loadtxt(est_file)
    est_t = est_data[:, 0]
    est_pos = est_data[:, 1:4]
    est_quat = est_data[:, 4:8] # qx, qy, qz, qw

    gt_t = np.loadtxt(gt_time_file)
    gt_poses = np.loadtxt(gt_pose_file)
    gt_pos = gt_poses[:, 0:3]
    gt_quat = gt_poses[:, 3:7] # qx, qy, qz, qw

    # 2. Interpolate Ground Truth to match Estimate Timestamps
    # We do this because the sensors and GT might not trigger at the exact same microsecond
    gt_pos_interp = np.zeros_like(est_pos)
    for i in range(3):
        gt_pos_interp[:, i] = interp1d(gt_t, gt_pos[:, i], fill_value="extrapolate")(est_t)

    # For rotation, we interpolate Euler angles (unwrapped to avoid jumps)
    gt_euler = R.from_quat(gt_quat).as_euler('xyz')
    gt_euler_unwrapped = np.unwrap(gt_euler, axis=0)
    
    gt_euler_interp = np.zeros((len(est_t), 3))
    for i in range(3):
        gt_euler_interp[:, i] = interp1d(gt_t, gt_euler_unwrapped[:, i], fill_value="extrapolate")(est_t)
    
    est_euler = R.from_quat(est_quat).as_euler('xyz')

    # 3. Calculate Errors (Absolute Trajectory Error - ATE)
    err_pos = est_pos - gt_pos_interp
    err_x, err_y, err_z = err_pos[:, 0], err_pos[:, 1], err_pos[:, 2]

    err_euler = wrap_angle_pi(est_euler - gt_euler_interp)
    err_roll, err_pitch, err_yaw = np.degrees(err_euler[:, 0]), np.degrees(err_euler[:, 1]), np.degrees(err_euler[:, 2])

    # 4. Compute RMSE
    metrics = {
        'rmse_x': calculate_rmse(err_x),
        'rmse_y': calculate_rmse(err_y),
        'rmse_z': calculate_rmse(err_z),
        'rmse_pos_3d': calculate_rmse(np.linalg.norm(err_pos, axis=1)),
        'rmse_roll': calculate_rmse(err_roll),
        'rmse_pitch': calculate_rmse(err_pitch),
        'rmse_yaw': calculate_rmse(err_yaw)
    }

    print("\n" + "="*40)
    print("      TRAJECTORY EVALUATION RESULTS")
    print("="*40)
    print(f"Translation RMSE (X): {metrics['rmse_x']:.3f} m")
    print(f"Translation RMSE (Y): {metrics['rmse_y']:.3f} m")
    print(f"Translation RMSE (Z): {metrics['rmse_z']:.3f} m")
    print(f"Total 3D Pos RMSE   : {metrics['rmse_pos_3d']:.3f} m")
    print("-" * 40)
    print(f"Rotation RMSE (Roll) : {metrics['rmse_roll']:.3f} deg")
    print(f"Rotation RMSE (Pitch): {metrics['rmse_pitch']:.3f} deg")
    print(f"Rotation RMSE (Yaw)  : {metrics['rmse_yaw']:.3f} deg")
    print("="*40 + "\n")

    # 5. Plotting
    if show_plots:
        time = est_t - est_t[0]
        fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        fig.suptitle('6-DoF Absolute Trajectory Error (ATE)', fontsize=16)

        # Plot Translation Errors
        axes[0, 0].plot(time, err_x, 'r', label=f'X Error (RMSE: {metrics["rmse_x"]:.2f}m)')
        axes[0, 0].set_ylabel('Meters'); axes[0, 0].legend(); axes[0, 0].grid(True)
        
        axes[1, 0].plot(time, err_y, 'g', label=f'Y Error (RMSE: {metrics["rmse_y"]:.2f}m)')
        axes[1, 0].set_ylabel('Meters'); axes[1, 0].legend(); axes[1, 0].grid(True)
        
        axes[2, 0].plot(time, err_z, 'b', label=f'Z Error (RMSE: {metrics["rmse_z"]:.2f}m)')
        axes[2, 0].set_ylabel('Meters'); axes[2, 0].set_xlabel('Time (s)'); axes[2, 0].legend(); axes[2, 0].grid(True)

        # Plot Rotation Errors
        axes[0, 1].plot(time, err_roll, 'r', linestyle='--', label=f'Roll Error (RMSE: {metrics["rmse_roll"]:.2f}°)')
        axes[0, 1].set_ylabel('Degrees'); axes[0, 1].legend(); axes[0, 1].grid(True)

        axes[1, 1].plot(time, err_pitch, 'g', linestyle='--', label=f'Pitch Error (RMSE: {metrics["rmse_pitch"]:.2f}°)')
        axes[1, 1].set_ylabel('Degrees'); axes[1, 1].legend(); axes[1, 1].grid(True)

        axes[2, 1].plot(time, err_yaw, 'b', linestyle='--', label=f'Yaw Error (RMSE: {metrics["rmse_yaw"]:.2f}°)')
        axes[2, 1].set_ylabel('Degrees'); axes[2, 1].set_xlabel('Time (s)'); axes[2, 1].legend(); axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()

    return metrics

if __name__ == '__main__':
    # You can easily run this from the command line for different files
    parser = argparse.ArgumentParser(description="Evaluate 6-DoF Trajectory Error")
    parser.add_argument('--est', type=str, default='/home/ajay/work/temp/aspen_run7/reve/gt_aided_odometry_final_traj.txt')
    parser.add_argument('--gt_poses', type=str, default='/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt')
    parser.add_argument('--gt_times', type=str, default='/home/ajay/work/temp/aspen_run7/groundtruth/timestamps.txt')
    parser.add_argument('--no_plot', action='store_true', help="Disable plotting for automated sweeps")
    
    args = parser.parse_args()
    evaluate_trajectory(args.est, args.gt_poses, args.gt_times, show_plots=not args.no_plot)