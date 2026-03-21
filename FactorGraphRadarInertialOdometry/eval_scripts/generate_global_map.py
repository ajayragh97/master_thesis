import os
import json
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import open3d as o3d

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_PATH = '../config.json'
# Trajectory to use as the backbone
TRAJECTORY_FILE = '/home/ajay/work/temp/aspen_run7/results/reve_icp_imu/skip_1000.txt'
IS_GROUND_TRUTH = False  

# NEW: Toggle to use the clean clouds instead of raw noise!
USE_FILTERED_CLOUDS = True

OUTPUT_PCD_NAME = '/home/ajay/work/temp/aspen_run7/results/reve_icp_imu/map_filtered_skip_1000.pcd'

def load_radar_bin(file_path):
    """ Loads the TI Radar binary file (x, y, z, snr, doppler) """
    points =[]
    with open(file_path, 'rb') as f:
        data = f.read()
        num_floats = len(data) // 4
        num_points = num_floats // 5
        unpacked = struct.unpack(f'{num_floats}f', data)
        for i in range(num_points):
            base = i * 5
            x, y, z, snr, doppler = unpacked[base:base+5]
            points.append([x, y, z, snr]) # Keeping SNR for visualization
    return np.array(points)

def read_tf_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    t = np.array([float(s) for s in lines[0].split()])
    q = np.array([float(s) for s in lines[1].split()]) # qx, qy, qz, qw
    return t, R.from_quat(q)

def build_global_map():
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    
    base_dir = cfg["dataset"]["base_dir"]
    sensor = cfg["dataset"]["sensor"]
    radar_times_file = os.path.join(base_dir, sensor, "pointclouds/timestamps.txt")
    
    # Select folder based on toggle
    if USE_FILTERED_CLOUDS:
        radar_dir = os.path.join(base_dir, sensor, "pointclouds/filtered_data")
        prefix = "filtered_pointcloud_"
        print("Using REVE-FILTERED static point clouds...")
    else:
        radar_dir = os.path.join(base_dir, sensor, "pointclouds/data")
        prefix = "radar_pointcloud_"
        print("Using RAW noisy point clouds...")
        
    t_base_radar, r_base_radar = read_tf_file(cfg["calibration"]["tf_radar_path"])
    
    print(f"Loading Trajectory: {TRAJECTORY_FILE}")
    traj_data = np.loadtxt(TRAJECTORY_FILE)
    
    traj_t = traj_data[:, 0]
    traj_pos = traj_data[:, 1:4]
    traj_quat = traj_data[:, 4:8]

    pos_interp = interp1d(traj_t, traj_pos, axis=0, bounds_error=False, fill_value="extrapolate")
    traj_euler = np.unwrap(R.from_quat(traj_quat).as_euler('xyz'), axis=0)
    euler_interp = interp1d(traj_t, traj_euler, axis=0, bounds_error=False, fill_value="extrapolate")

    radar_times = np.loadtxt(radar_times_file)
    global_points = []
    global_intensities =[]

    print("Stitching Point Cloud Map... This may take a moment.")
    for i, t_radar in enumerate(radar_times):
        bin_path = os.path.join(radar_dir, f"{prefix}{i}.bin")
        if not os.path.exists(bin_path):
            continue
            
        if t_radar < traj_t[0] or t_radar > traj_t[-1]:
            continue

        p_global_body = pos_interp(t_radar)
        r_global_body = R.from_euler('xyz', euler_interp(t_radar))

        pts_raw = load_radar_bin(bin_path)
        if len(pts_raw) < 5:
            continue
            
        pts_sensor = pts_raw[:, 0:3]
        snr = pts_raw[:, 3]
        
        # Sensor -> Body -> Global
        pts_body = r_base_radar.apply(pts_sensor) + t_base_radar
        pts_global = r_global_body.apply(pts_body) + p_global_body
        
        global_points.append(pts_global)
        global_intensities.append(snr)

        if i % 100 == 0:
            print(f"Processed {i} / {len(radar_times)} frames...")

    all_points = np.vstack(global_points)
    all_snr = np.concatenate(global_intensities)
    
    snr_norm = (all_snr - np.min(all_snr)) / (np.max(all_snr) - np.min(all_snr))
    # Import matplotlib locally to avoid backend issues on some headless servers
    import matplotlib.pyplot as plt 
    colors = plt.cm.jet(snr_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Total Points in Map: {len(all_points)}")
    
    print("Downsampling to 20cm voxels to save space...")
    pcd = pcd.voxel_down_sample(voxel_size=0.2)
    
    o3d.io.write_point_cloud(OUTPUT_PCD_NAME, pcd)
    print(f"Successfully saved Global Map to {OUTPUT_PCD_NAME}")

if __name__ == "__main__":
    build_global_map()