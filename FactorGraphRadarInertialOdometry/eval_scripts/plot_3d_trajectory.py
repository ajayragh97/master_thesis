import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# ==========================================
# Configuration / File Paths
# ==========================================
ESTIMATED_TRAJ_FILE = '/home/ajay/work/temp/aspen_run7/reve/gt_aided_odometry_final_traj.txt'
GT_POSES_FILE       = '/home/ajay/work/temp/aspen_run7/groundtruth/groundtruth_poses.txt'

def plot_3d_covariance_ellipsoid(ax, pos, cov_matrix, n_std=3.0, **kwargs):
    """
    Generates and plots a 3D error ellipsoid.
    n_std = 3.0 represents ~99% confidence interval.
    """
    # 1. Generate a unit sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Pack into a 3xN array
    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten()))

    # 2. Eigen decomposition to get rotation and scaling
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Prevent numerical issues with tiny negative eigenvalues
    eigenvalues = np.abs(eigenvalues) 
    
    # Scale by the standard deviation and eigenvalues
    radii = n_std * np.sqrt(eigenvalues)
    
    # Transform: Scale then Rotate
    transformation = eigenvectors @ np.diag(radii)
    ellipsoid_points = transformation @ sphere_points

    # 3. Translate to the vehicle's position
    x_ell = ellipsoid_points[0, :].reshape(x.shape) + pos[0]
    y_ell = ellipsoid_points[1, :].reshape(y.shape) + pos[1]
    z_ell = ellipsoid_points[2, :].reshape(z.shape) + pos[2]

    # Plot the surface
    ax.plot_surface(x_ell, y_ell, z_ell, **kwargs)

def main():
    # 1. Load Estimated Data
    print("Loading Estimate Data...")
    if not os.path.exists(ESTIMATED_TRAJ_FILE):
        print(f"File not found: {ESTIMATED_TRAJ_FILE}")
        return

    est_data = np.loadtxt(ESTIMATED_TRAJ_FILE)
    
    t_est = est_data[:, 0]
    x_est, y_est, z_est = est_data[:, 1], est_data[:, 2], est_data[:, 3]
    q_est = est_data[:, 4:8] # qx, qy, qz, qw
    
    # Extract 3x3 Translation Covariance 
    c_xx, c_xy, c_xz = est_data[:, 8], est_data[:, 9], est_data[:, 10]
    c_yy, c_yz, c_zz = est_data[:, 11], est_data[:, 12], est_data[:, 13]

    # Calculate Forward Vectors (assuming robot forward is local +X axis)
    rot_est = R.from_quat(q_est)
    forward_vectors = rot_est.apply([1.0, 0.0, 0.0]) # Rotate the unit X vector
    u_est, v_est, w_est = forward_vectors[:, 0], forward_vectors[:, 1], forward_vectors[:, 2]

    # 2. Load Ground Truth
    print("Loading Ground Truth...")
    gt_data = np.loadtxt(GT_POSES_FILE)
    x_gt, y_gt, z_gt = gt_data[:, 0], gt_data[:, 1], gt_data[:, 2]
    
    # ==========================================
    # PLOTTING
    # ==========================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Ground Truth Path
    ax.plot(x_gt, y_gt, z_gt, color='gray', linestyle='--', linewidth=2.5, label='Ground Truth')
    
    # Plot Estimated Path
    ax.plot(x_est, y_est, z_est, color='blue', linestyle='-', linewidth=2, label='Factor Graph Estimate')

    print("Generating 3D Covariance Ellipsoids and Quivers...")
    # Plot Covariance Ellipsoids and Orientation Arrows
    # Step to prevent the plot from becoming a solid red block
    STEP = 1000 
    
    for i in range(0, len(t_est), STEP):
        # Build the 3x3 covariance matrix
        cov3d = np.array([
            [c_xx[i], c_xy[i], c_xz[i]],
            [c_xy[i], c_yy[i], c_yz[i]],
            [c_xz[i], c_yz[i], c_zz[i]]
        ])
        
        pos3d = np.array([x_est[i], y_est[i], z_est[i]])
        
        # Draw 3-Sigma Error Ellipsoid (~99% confidence)
        plot_3d_covariance_ellipsoid(ax, pos3d, cov3d, n_std=3.0, 
                                     color='red', alpha=0.15, edgecolor='none')
        
        # Draw 3D Orientation Arrow (Quiver)
        # Length is scaled to make the arrow visible
        ax.quiver(x_est[i], y_est[i], z_est[i], 
                  u_est[i], v_est[i], w_est[i], 
                  color='black', length=0.5, normalize=True, alpha=0.8)

    # Add marker for Start
    ax.scatter(x_est[0], y_est[0], z_est[0], c='green', marker='^', s=200, zorder=5, label='Start')

    # Formatting
    ax.set_title('3D Trajectory with 3-$\sigma$ Covariance & Orientation', fontsize=16)
    ax.set_xlabel('X (meters)', labelpad=10)
    ax.set_ylabel('Y (meters)', labelpad=10)
    ax.set_zlabel('Z (meters)', labelpad=10)
    ax.legend(loc='upper right')
    
    # Set equal aspect ratio for 3D plot (prevents distortion of spheres into ovals)
    try:
        # For newer matplotlib versions
        ax.set_box_aspect([
            np.ptp(ax.get_xlim()), 
            np.ptp(ax.get_ylim()), 
            np.ptp(ax.get_zlim())
        ])
    except AttributeError:
        pass

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()