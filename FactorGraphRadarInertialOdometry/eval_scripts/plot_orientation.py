import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import os

# ==========================================
# Configuration / File Paths
# ==========================================
ESTIMATED_TRAJ_FILE = '/home/ajay/work/temp/aspen_run7/reve/single_chip_odometry_full_optimized.txt'
GT_POSES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'
GT_TIMES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/timestamps.txt'

def wrap_angle_degrees(angle):
    """ Wraps an angle in degrees to the [-180, 180] range """
    return (angle + 180) % 360 - 180

def main():
    # 1. Load Estimated Data
    print("Loading Estimate Data...")
    if not os.path.exists(ESTIMATED_TRAJ_FILE):
        print(f"Error: File not found {ESTIMATED_TRAJ_FILE}")
        return

    est_data = np.loadtxt(ESTIMATED_TRAJ_FILE)
    est_t = est_data[:, 0]
    est_quat = est_data[:, 4:8] # qx, qy, qz, qw

    # 2. Load Ground Truth Data
    print("Loading Ground Truth Data...")
    gt_t = np.loadtxt(GT_TIMES_FILE)
    gt_poses = np.loadtxt(GT_POSES_FILE)
    gt_quat = gt_poses[:, 3:7] # qx, qy, qz, qw

    # 3. Convert Quaternions to Euler Angles (Roll, Pitch, Yaw)
    # Using 'xyz' convention: X=Roll, Y=Pitch, Z=Yaw
    est_euler = R.from_quat(est_quat).as_euler('xyz', degrees=False)
    gt_euler = R.from_quat(gt_quat).as_euler('xyz', degrees=False)

    # Unwrap angles in radians to prevent 2*PI jumps, then convert to degrees
    est_euler_deg = np.degrees(np.unwrap(est_euler, axis=0))
    gt_euler_deg = np.degrees(np.unwrap(gt_euler, axis=0))

    # 4. Interpolate Ground Truth to match Estimate timestamps
    print("Interpolating and calculating errors...")
    gt_interp_roll = interp1d(gt_t, gt_euler_deg[:, 0], fill_value="extrapolate")(est_t)
    gt_interp_pitch = interp1d(gt_t, gt_euler_deg[:, 1], fill_value="extrapolate")(est_t)
    gt_interp_yaw = interp1d(gt_t, gt_euler_deg[:, 2], fill_value="extrapolate")(est_t)

    # 5. Calculate Errors (Wrapped to [-180, 180] to handle wrap-around bugs)
    err_roll = wrap_angle_degrees(est_euler_deg[:, 0] - gt_interp_roll)
    err_pitch = wrap_angle_degrees(est_euler_deg[:, 1] - gt_interp_pitch)
    err_yaw = wrap_angle_degrees(est_euler_deg[:, 2] - gt_interp_yaw)

    # Calculate RMSE (Root Mean Square Error) for display
    rmse_roll = np.sqrt(np.mean(err_roll**2))
    rmse_pitch = np.sqrt(np.mean(err_pitch**2))
    rmse_yaw = np.sqrt(np.mean(err_yaw**2))

    # Time array relative to 0
    time = est_t - est_t[0]

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('3D Orientation Evaluation: Factor Graph vs Ground Truth', fontsize=16)

    # ---- ROLL ----
    # Absolute
    axes[0, 0].plot(time, gt_interp_roll, 'k--', linewidth=2.5, label='Ground Truth')
    axes[0, 0].plot(time, est_euler_deg[:, 0], 'b-', alpha=0.8, linewidth=2, label='Estimate')
    axes[0, 0].set_title('Roll Angle (X-Axis)')
    axes[0, 0].set_ylabel('Degrees')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)

    # Error
    axes[0, 1].plot(time, err_roll, 'r-', alpha=0.8, label=f'RMSE: {rmse_roll:.3f}°')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Roll Error')
    axes[0, 1].set_ylabel('Error (Degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)

    # ---- PITCH ----
    # Absolute
    axes[1, 0].plot(time, gt_interp_pitch, 'k--', linewidth=2.5, label='Ground Truth')
    axes[1, 0].plot(time, est_euler_deg[:, 1], 'g-', alpha=0.8, linewidth=2, label='Estimate')
    axes[1, 0].set_title('Pitch Angle (Y-Axis)')
    axes[1, 0].set_ylabel('Degrees')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=':', alpha=0.7)

    # Error
    axes[1, 1].plot(time, err_pitch, 'r-', alpha=0.8, label=f'RMSE: {rmse_pitch:.3f}°')
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Pitch Error')
    axes[1, 1].set_ylabel('Error (Degrees)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)

    # ---- YAW ----
    # Absolute
    axes[2, 0].plot(time, gt_interp_yaw, 'k--', linewidth=2.5, label='Ground Truth')
    axes[2, 0].plot(time, est_euler_deg[:, 2], 'm-', alpha=0.8, linewidth=2, label='Estimate')
    axes[2, 0].set_title('Yaw Angle (Z-Axis)')
    axes[2, 0].set_xlabel('Time (seconds)')
    axes[2, 0].set_ylabel('Degrees')
    axes[2, 0].legend()
    axes[2, 0].grid(True, linestyle=':', alpha=0.7)

    # Error
    axes[2, 1].plot(time, err_yaw, 'r-', alpha=0.8, label=f'RMSE: {rmse_yaw:.3f}°')
    axes[2, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2, 1].set_title('Yaw Error')
    axes[2, 1].set_xlabel('Time (seconds)')
    axes[2, 1].set_ylabel('Error (Degrees)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == '__main__':
    main()