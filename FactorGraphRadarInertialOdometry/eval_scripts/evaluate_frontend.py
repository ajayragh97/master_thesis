import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ==========================================
# PATHS
# ==========================================
EVAL_CSV_FILE = '/home/ajay/work/temp/aspen_run7/reve/sensor_eval.csv'
GT_POSES_FILE = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'
GT_TIMES_FILE = '/home/ajay/work/temp/aspen_run7/groundtruth/timestamps.txt'

def main():
    # 1. Load Data
    data = np.genfromtxt(EVAL_CSV_FILE, delimiter=',', skip_header=1)
    t_est = data[:, 0]
    
    # Radar Data
    reve_v = data[:, 1:4] # vx, vy, vz
    
    # IMU Data
    imu_a = data[:, 5:8]  # ax, ay, az
    imu_w = data[:, 8:11] # wx, wy, wz

    # Load GT
    gt_t = np.loadtxt(GT_TIMES_FILE)
    gt_poses = np.loadtxt(GT_POSES_FILE)
    gt_pos = gt_poses[:, 0:3] # x, y, z
    gt_quat = gt_poses[:, 3:7] # qx, qy, qz, qw

    # 2. Derive GT Body Velocity & Acceleration
    print("Deriving GT Dynamics...")
    
    # Compute Global Velocity (Central Difference)
    dt_gt = np.gradient(gt_t)
    v_global = np.gradient(gt_pos, axis=0) / dt_gt[:, None]
    
    # Smooth Global Velocity (Differentiation is noisy)
    v_global = gaussian_filter1d(v_global, sigma=5, axis=0)
    
    # Compute Global Acceleration
    a_global = np.gradient(v_global, axis=0) / dt_gt[:, None]
    a_global = gaussian_filter1d(a_global, sigma=5, axis=0)
    
    # Add Gravity to Global Accel (IMU measures Proper Acceleration)
    # Assuming Z-up world frame, gravity is [0, 0, -9.81]
    # So proper acceleration = a_kinematic - g = a_kinematic + 9.81
    a_global[:, 2] += 9.81

    # Compute Body Frame Transforms
    r_gt = R.from_quat(gt_quat)
    r_inv = r_gt.inv()
    
    # Transform to Body Frame
    v_body_gt = r_inv.apply(v_global)
    a_body_gt = r_inv.apply(a_global)
    
    # Compute Angular Velocity (Body Frame)
    # w = 2 * dq/dt * q_conjugate (approximation via rotation vectors)
    # Simpler: diff Euler angles? No, gimbal lock.
    # Robust way: Relative rotation between steps
    rot_vecs = (r_gt[:-1].inv() * r_gt[1:]).as_rotvec()
    w_body_gt = np.zeros_like(v_body_gt)
    w_body_gt[:-1] = rot_vecs / dt_gt[:-1, None]
    w_body_gt = gaussian_filter1d(w_body_gt, sigma=5, axis=0)

    # 3. Interpolate GT to Estimator Time
    def interp_vec(gt_vals):
        f = interp1d(gt_t, gt_vals, axis=0, bounds_error=False, fill_value="extrapolate")
        return f(t_est)

    v_body_gt_sync = interp_vec(v_body_gt)
    a_body_gt_sync = interp_vec(a_body_gt)
    w_body_gt_sync = interp_vec(w_body_gt)

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    time = t_est - t_est[0]

    # --- ROW 1: VELOCITY (Radar vs GT) ---
    axes[0,0].plot(time, v_body_gt_sync[:,0], 'k--', label='GT')
    axes[0,0].plot(time, reve_v[:,0], 'b', alpha=0.7, label='REVE')
    axes[0,0].set_title('Velocity X (Surge)')
    
    axes[0,1].plot(time, v_body_gt_sync[:,1], 'k--', label='GT')
    axes[0,1].plot(time, reve_v[:,1], 'b', alpha=0.7, label='REVE')
    axes[0,1].set_title('Velocity Y (Sway)')

    axes[0,2].plot(time, v_body_gt_sync[:,2], 'k--', label='GT')
    axes[0,2].plot(time, reve_v[:,2], 'b', alpha=0.7, label='REVE')
    axes[0,2].set_title('Velocity Z (Heave)')

    # --- ROW 2: ACCELERATION (IMU vs GT) ---
    axes[1,0].plot(time, a_body_gt_sync[:,0], 'k--', label='GT')
    axes[1,0].plot(time, imu_a[:,0], 'r', alpha=0.7, label='IMU')
    axes[1,0].set_title('Accel X')

    axes[1,1].plot(time, a_body_gt_sync[:,1], 'k--', label='GT')
    axes[1,1].plot(time, imu_a[:,1], 'r', alpha=0.7, label='IMU')
    axes[1,1].set_title('Accel Y')

    axes[1,2].plot(time, a_body_gt_sync[:,2], 'k--', label='GT')
    axes[1,2].plot(time, imu_a[:,2], 'r', alpha=0.7, label='IMU')
    axes[1,2].set_title('Accel Z (Gravity)')

    # --- ROW 3: GYRO (IMU vs GT) ---
    axes[2,0].plot(time, w_body_gt_sync[:,0], 'k--', label='GT')
    axes[2,0].plot(time, imu_w[:,0], 'g', alpha=0.7, label='IMU')
    axes[2,0].set_title('Gyro X (Roll Rate)')

    axes[2,1].plot(time, w_body_gt_sync[:,1], 'k--', label='GT')
    axes[2,1].plot(time, imu_w[:,1], 'g', alpha=0.7, label='IMU')
    axes[2,1].set_title('Gyro Y (Pitch Rate)')

    axes[2,2].plot(time, w_body_gt_sync[:,2], 'k--', label='GT')
    axes[2,2].plot(time, imu_w[:,2], 'g', alpha=0.7, label='IMU')
    axes[2,2].set_title('Gyro Z (Yaw Rate)')

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()