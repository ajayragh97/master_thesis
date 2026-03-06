#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

#include "radar_common/Types.h"
#include "radar_common/SystemConfig.h"
#include "radar_common/Utils.h"
#include "radar_velocity_estimation/Estimator.h"
#include "radar_velocity_estimation/BodyFrameCorrector.h"
#include "radar_registration/RadarICP.h"
#include "factor_graph_optimization/GraphOptimizer.h"

namespace fs = std::filesystem;
using namespace radar::common;
using namespace radar::velocity;
using namespace radar::registration;
using namespace radar::optimization;

int main(int argc, char** argv)
{
    std::string config_path = "../config.json"; // Default config path
    if (argc > 1) config_path = argv[1]; // Allow custom config path as command-line argument

    radar::common::SystemConfig cfg;
    try
    {
        cfg = radar::common::SystemConfig::load(config_path);
        std::cout << "Configuration loaded successfully from: " << config_path << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading configuration: " << e.what() << std::endl;
        return -1;
    }

    // Setup paths from config
    std::string base_dir = cfg.dataset.base_dir;
    std::string calib_dir = cfg.dataset.calib_dir;
    std::string data_dir = base_dir  + cfg.dataset.sensor + "/pointclouds/data/";
    std::string stamp_path = base_dir + cfg.dataset.sensor + "/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    std::string full_optimized_output_path = base_dir + cfg.dataset.output_filename + "_full_optimized.txt";
    std::string real_time_optimized_output_path = base_dir + cfg.dataset.output_filename + "_real_time_optimized.txt";

    // Load data
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    std::vector<ImuData> all_imu_data = loadImuDataVector(imu_data_path, imu_stamp_path);

    double TARGET_DT = 1.0 / cfg.graph.keyframe_rate; // Desired time interval between graph nodes 

    // Sanity check for timestamps to ensure their units
    if (timestamps.size() > 1) {
        double avg_dt = (timestamps.back() - timestamps.front()) / (timestamps.size() - 1);
        
        std::cout << "Detected Average Radar dt: " << avg_dt << std::endl;

        if (avg_dt > 1000.0) {
            std::cerr << "CRITICAL ERROR: Timestamps appear to be in MICROSECONDS or NANOSECONDS." << std::endl;
            std::cerr << "Please convert them to SECONDS before running this optimization." << std::endl;
            return -1;
        }
        
        if (avg_dt > 10.0) {
             std::cerr << "WARNING: Frame rate is very low (< 0.1 Hz). Check your data." << std::endl;
        }

        // Check if TARGET_DT is reasonable
        if (TARGET_DT >= avg_dt) {
             std::cerr << "WARNING: Your Fixed-Rate DT (" << TARGET_DT 
                       << "s) is larger than the Sensor DT (" << avg_dt 
                       << "s). This defeats the purpose of fixed-rate optimization!" << std::endl;
        }
    }

    if (timestamps.empty() || all_imu_data.empty())
    {
        std::cerr << "Missing imu data or timestamps" << std::endl;
        return -1;
    }

    // Load extrinsic calibration
    BodyFrameCorrector corrector;
    Transform t_radar, t_imu; 
    if (!corrector.loadTransform(cfg.tf_radar_path, t_radar) || 
        !corrector.loadTransform(cfg.tf_imu_path, t_imu)) 
    {
        std::cerr << "Error loading transforms!" << std::endl;
        return -1;
    }
    corrector.setTransform(t_radar, t_imu);
    cfg.graph.T_base_imu = t_imu;
    corrector.loadImuData(imu_data_path, imu_stamp_path); // for omega interpolation

    Eigen::Matrix4d T_base_radar = corrector.getRadarToBaseTransform();
    Eigen::Matrix4d T_radar_base = T_base_radar.inverse();

    // Init modules
    RadarEgoEstimator vel_estimator(cfg.reve);
    RadarICP icp_solver(cfg.icp);
    GraphOptimizer optimizer(cfg.graph);

    optimizer.set2DMode(cfg.graph.use_2d_mode); // Set to false to optimize full 3D pose (x,y,z,roll,pitch,yaw)
    

    // Determine start frame based on static duration (if specified)
    size_t start_idx = 0;
    if (cfg.dataset.static_duration > 0.0)
    {
        for (size_t i = 0; i < timestamps.size(); ++i)
        {
            if (timestamps[i] - timestamps[0] > cfg.dataset.static_duration)
            {
                start_idx = i;
                break;
            }
        }
    }
    std::cout << "Skipping first " << start_idx << " frames based on static duration of " << cfg.dataset.static_duration << " seconds." << std::endl;

    
    // initialize graph


    std::ofstream outFile(real_time_optimized_output_path);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << real_time_optimized_output_path << std::endl;
        return 0;
    }
    outFile << std::fixed << std::setprecision(6);

    gtsam::Pose3 init_pose(gtsam::Rot3::Quaternion(cfg.icp.start_qw, cfg.icp.start_qx, cfg.icp.start_qy, cfg.icp.start_qz),
                           gtsam::Point3(cfg.icp.start_tx, cfg.icp.start_ty, cfg.icp.start_tz));
    
    gtsam::Vector3 init_velocity = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias init_bias;

    optimizer.initialize(init_pose, init_velocity, init_bias, timestamps[start_idx]);

    outFile << timestamps[start_idx] << " "
            << init_pose.translation().x() << " "
            << init_pose.translation().y() << " "
            << init_pose.translation().z() << " "
            << cfg.icp.start_qx << " " 
            << cfg.icp.start_qy << " " 
            << cfg.icp.start_qz << " " 
            << cfg.icp.start_qw << "\n";

    // State tracking
    double last_node_timestamp = timestamps[start_idx];
    size_t current_imu_idx = 0;

    // fast forward IMU to start
    while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp < last_node_timestamp) 
    {
        current_imu_idx++;
    }

    PointCloud prev_cloud;
    std::cout << "Starting Fixed-Rate Optimization (Target: " << (cfg.graph.keyframe_rate) << "Hz)..." << std::endl;

    // Keeping track of last known IMU reading for gap filling
    ImuData last_known_imu;
    if (!all_imu_data.empty()) last_known_imu = all_imu_data[current_imu_idx];

    double last_valid_radar_time = timestamps[start_idx];

    // Main Loop
    for (size_t i = start_idx; i < timestamps.size(); ++i)
    {
        // 1. Process radar data
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";
        if (!fs::exists(bin_path)) continue;
        PointCloud raw_cloud = loadPointCloud(bin_path);
        if (raw_cloud.size() < 10) continue; // Skip if not enough points

        // 2. Estimate velocity
        VelocityEstimate vel_estimate_sensor = vel_estimator.estimate(raw_cloud, false);

        // 3. Filter dynamic points if velocity estimate is valid
        PointCloud static_cloud;
        if (vel_estimate_sensor.success)
        {
            static_cloud = filterDynamicPoints(raw_cloud, vel_estimate_sensor.linear_velocity, cfg.icp.dynamic_point_vel_thresh);
        } 
        else 
        {
            static_cloud = raw_cloud; // If velocity estimate failed, use raw cloud for ICP (less accurate)
        }

        // If this is the first frame, initialize prev_cloud and skip optimization (no ICP or graph factors to add yet)
        if (i == start_idx)
        {
            prev_cloud = static_cloud;
            continue;
        }

        // ==========================================
        // Fixed-Rate Graph Node Creation
        // ==========================================

        double target_radar_time = timestamps[i];

        // If the next virtual node would be closer than 2ms to the radar frame, STOP.
        // We will just jump straight to the radar frame.
        double time_buffer = 0.002; 

        // creating intermediate nodes at fixed rate between last node and current timestamp
        while (last_node_timestamp + TARGET_DT < target_radar_time - time_buffer)
        {
            double next_node_timestamp = last_node_timestamp + TARGET_DT;

            OptimizationFrame virtual_frame;
            virtual_frame.timestamp = next_node_timestamp;
            virtual_frame.has_reve_velocity = false; // No velocity measurement for virtual frame
            virtual_frame.has_delta_yaw = false; // No ICP measurement for virtual frame

            // Get IMU slice for this virtual frame
            while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp <= next_node_timestamp)
            {
                if (all_imu_data[current_imu_idx].timestamp > last_node_timestamp) // Only add IMU data that is after the last node timestamp
                {
                    virtual_frame.imu_measurements.push_back(all_imu_data[current_imu_idx]);

                    // Update our memory of "current" motion
                    last_known_imu = all_imu_data[current_imu_idx];
                }
                current_imu_idx++;
            }


            // Add virtual Node
            if (!virtual_frame.imu_measurements.empty()) // Only add node if we have IMU data to integrate
            {
                optimizer.addFrame(virtual_frame);
            }
            last_node_timestamp = next_node_timestamp;
        }

        // ==========================================
        // Adding Radar Node at current timestamp
        // ==========================================
        OptimizationFrame radar_frame;
        radar_frame.timestamp = target_radar_time;

        // Fill remaining IMU data for the current radar frame
        while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp <= target_radar_time)
        {
            if (all_imu_data[current_imu_idx].timestamp > last_node_timestamp) // Only add IMU data that is after the last node timestamp (to avoid duplicates)
            {
                radar_frame.imu_measurements.push_back(all_imu_data[current_imu_idx]);

                // Update our memory of "current" motion
                last_known_imu = all_imu_data[current_imu_idx];
            }
            current_imu_idx++;
        }

        // --- Handling data gaps for radar frame ---
        if (radar_frame.imu_measurements.empty()) {
             ImuData synthetic_imu = last_known_imu;
             synthetic_imu.timestamp = target_radar_time;
             radar_frame.imu_measurements.push_back(synthetic_imu);
        }

        // Populate REVE velocity if available
        if (vel_estimate_sensor.success)
        {
            Eigen::Vector3d vel_estimate_body = corrector.correctVelocity(vel_estimate_sensor.linear_velocity, target_radar_time);
            radar_frame.reve_velocity_body.linear_velocity = vel_estimate_body;
            radar_frame.reve_velocity_body.inlier_ratio = vel_estimate_sensor.inlier_ratio;
            radar_frame.reve_velocity_body.covariance = vel_estimate_sensor.covariance;
            radar_frame.has_reve_velocity = true;
        }

        // Estimate delta yaw using ICP between prev_cloud and current static_cloud
        double dt_radar = timestamps[i] - last_valid_radar_time;
        Eigen::Vector3d T_guess_sensor = -vel_estimate_sensor.linear_velocity * dt_radar; // Predict translation in sensor frame using REVE velocity and time gap (assuming constant velocity model) - this is our initial guess for ICP. Even if REVE fails, this will just be zero, so ICP will still run but with a worse initial guess.
        Eigen::Matrix4d T_guess_icp = Eigen::Matrix4d::Identity();
        T_guess_icp.block<3,1>(0,3) = T_guess_sensor;

        Eigen::Matrix4d T_icp_sensor  = icp_solver.compute(static_cloud, prev_cloud, T_guess_icp, cfg.graph.use_2d_mode);
        Eigen::Matrix4d M_body = T_base_radar * T_icp_sensor * T_radar_base; // Convert to Body Frame

        radar_frame.delta_yaw = atan2(M_body(1,0), M_body(0,0));
        radar_frame.delta_yaw_fitness = icp_solver.getFitnessScore();
        radar_frame.has_delta_yaw = true;

        // ===============================
        // Debug output for current frame
        // ===============================
        if (i % 20 == 0) 
        { 
            std::cout << "--- Frame " << i << " ---" << std::endl;
            std::cout << "1. REVE Vel X (Body): " << radar_frame.reve_velocity_body.linear_velocity.x() << " m/s" << std::endl;
            
            if (!radar_frame.imu_measurements.empty()) 
            {
                std::cout << "2. IMU Accel X:       " << radar_frame.imu_measurements.back().linear_acceleration.x() << " m/s^2" << std::endl;
                std::cout << "3. IMU Gyro Z (Yaw):  " << radar_frame.imu_measurements.back().angular_velocity.z() * dt_radar << " rad" << std::endl;
            }
            std::cout << "4. ICP Delta Yaw:     " << radar_frame.delta_yaw << " rad" << std::endl;
            std::cout << "-----------------------" << std::endl;
        }
// ===========================================================

        // Add radar frame as a node in the graph
        optimizer.addFrame(radar_frame);

        // Update the node trackers
        last_node_timestamp = target_radar_time;
        prev_cloud = static_cloud;
        last_valid_radar_time = timestamps[i];

        // Write optimized pose to file
        gtsam::Pose3 optimized_pose = optimizer.getCurrentPose();
        outFile << last_node_timestamp << " "
                << optimized_pose.translation().x() << " "
                << optimized_pose.translation().y() << " "
                << optimized_pose.translation().z() << " "
                << optimized_pose.rotation().toQuaternion().x() << " "
                << optimized_pose.rotation().toQuaternion().y() << " "
                << optimized_pose.rotation().toQuaternion().z() << " "
                << optimized_pose.rotation().toQuaternion().w() << "\n";    

        // Progress output
        if (i % 100 == 0)
        {
            std::cout << "Processed frame " << i << "/" << timestamps.size() 
                      << " (Time: " << std::fixed << std::setprecision(2) << target_radar_time 
                      << "s, REVE Inlier Ratio: " << radar_frame.reve_velocity_body.inlier_ratio 
                      << "%, ICP Fitness: " << radar_frame.delta_yaw_fitness << ")" << std::endl;
        }
        
    }

    outFile.close();

    std::cout << "Fixed-Rate Optimization complete. " << std::endl;
    std::cout << "Real-time optimized trajectory saved to: " << real_time_optimized_output_path << std::endl;

    // Save final trajectory
    optimizer.saveFullTrajectory(full_optimized_output_path);
    return 0;
}

