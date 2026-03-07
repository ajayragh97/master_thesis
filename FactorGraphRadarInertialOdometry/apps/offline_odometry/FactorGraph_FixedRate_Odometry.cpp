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
    std::string eval_path = base_dir + "reve/sensor_eval.csv";
    std::ofstream eval_file(eval_path);
    // Log Radar (Vx, Vy, Vz, dYaw) and IMU (Ax, Ay, Az, Wx, Wy, Wz)
    eval_file << "timestamp,reve_vx,reve_vy,reve_vz,icp_dyaw,imu_ax,imu_ay,imu_az,imu_wx,imu_wy,imu_wz,dt\n";

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
    size_t start_idx = 0;

    
    // initialize graph
    std::ofstream outFile(real_time_optimized_output_path);
    if (!outFile.is_open()) 
    {
        std::cerr << "Error: Could not open file for writing: " << real_time_optimized_output_path << std::endl;
        return 0;
    }
    outFile << std::fixed << std::setprecision(6);

    //--- Static Bias Calibration (at startup) ---
    std::cout << "Calibrating IMU Biases from static period" << std::endl;
    std::vector<Eigen::Vector3d> acc_buffer, gyro_buffer;

    for (size_t i = 0; i < all_imu_data.size(); ++i)
    {
        if (all_imu_data[i].timestamp - all_imu_data[0].timestamp < cfg.dataset.static_duration)
        {
            acc_buffer.push_back(all_imu_data[i].linear_acceleration);
            gyro_buffer.push_back(all_imu_data[i].angular_velocity);
        }
        else
        {
            break;
        }
    }   

    // Compute mean bias
    Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();
    if (!acc_buffer.empty() && !gyro_buffer.empty())
    {
        for (const auto& acc : acc_buffer) 
        {
            mean_acc += acc;
        }

        for (const auto& gyro : gyro_buffer)
        {
            mean_gyro += gyro;
        }

        mean_acc /= acc_buffer.size();
        mean_gyro /= gyro_buffer.size();
    }

    double init_roll = atan2(mean_acc.y(), mean_acc.z());
    double init_pitch = atan2(-mean_acc.x(), sqrt(pow(mean_acc.y(), 2) + pow(mean_acc.z(), 2)));
    gtsam::Rot3 config_rot = gtsam::Rot3::Quaternion(cfg.icp.start_qw, cfg.icp.start_qx, cfg.icp.start_qy, cfg.icp.start_qz);
    double init_yaw = config_rot.yaw();
    
    // creating gravity aligned initial pose
    gtsam::Rot3 aligned_rot = gtsam::Rot3::Ypr(init_yaw, init_pitch, init_roll);
    gtsam::Pose3 init_pose(aligned_rot, gtsam::Point3(cfg.icp.start_tx, cfg.icp.start_ty, cfg.icp.start_tz));

    gtsam::Vector3 gravity_global(0, 0, cfg.graph.gravity);
    gtsam::Vector3 gravity_sensor = aligned_rot.inverse() * gravity_global;
    gtsam::imuBias::ConstantBias init_bias(mean_acc - gravity_sensor, mean_gyro);

    gtsam::Vector3 init_velocity = gtsam::Vector3::Zero();

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
    double last_valid_radar_time = timestamps[start_idx];
    size_t current_imu_idx = 0;

    // fast forward IMU to start
    while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp < last_valid_radar_time) 
    {
        current_imu_idx++;
    }

    PointCloud prev_cloud;

    // Keeping track of last known IMU reading for gap filling
    ImuData last_known_imu;
    if (!all_imu_data.empty()) last_known_imu = all_imu_data[current_imu_idx];

    // Main Loop
    for (size_t i = start_idx; i < timestamps.size(); ++i)
    {
        // 1. Process radar data
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";
        if (!fs::exists(bin_path)) continue;
        PointCloud raw_cloud = loadPointCloud(bin_path);
        if (raw_cloud.size() < 10) continue; // Skip if not enough points

        double current_radar_time = timestamps[i];
        double dt_radar = current_radar_time - last_valid_radar_time;

        // 2. Estimate velocity
        VelocityEstimate vel_estimate_sensor = vel_estimator.estimate(raw_cloud, false);

        // 3. Filter dynamic points if velocity estimate is valid
        PointCloud static_cloud = vel_estimate_sensor.success ? filterDynamicPoints(raw_cloud, vel_estimate_sensor.linear_velocity, cfg.icp.dynamic_point_vel_thresh) : raw_cloud;

        // If this is the first frame, initialize prev_cloud and skip optimization (no ICP or graph factors to add yet)
        if (i == start_idx)
        {
            prev_cloud = static_cloud;
            continue;
        }

        // 4. Create radar frame and populate IMU measurements for this frame
        OptimizationFrame radar_frame;
        radar_frame.timestamp = current_radar_time;

        // Gather all IMU measurements since last valid radar frame
        while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp <= current_radar_time)
        {
            if (all_imu_data[current_imu_idx].timestamp > last_valid_radar_time) 
            {
                radar_frame.imu_measurements.push_back(all_imu_data[current_imu_idx]);
            }
            current_imu_idx++;
        }

        // 5. Assign radar factors
        // Populate REVE velocity if available
        if (vel_estimate_sensor.success)
        {
            Eigen::Vector3d vel_estimate_body = corrector.correctVelocity(vel_estimate_sensor.linear_velocity, current_radar_time);
            radar_frame.reve_velocity_body.linear_velocity = vel_estimate_body;
            radar_frame.reve_velocity_body.inlier_ratio = vel_estimate_sensor.inlier_ratio;
            radar_frame.reve_velocity_body.covariance = vel_estimate_sensor.covariance;
            radar_frame.has_reve_velocity = true;
        }

        // Estimate delta yaw using ICP between prev_cloud and current static_cloud
        Eigen::Vector3d T_guess_sensor = vel_estimate_sensor.linear_velocity * dt_radar; 
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

        // Log evaluation data
        // Get the last IMU reading for this frame (as a sanity check sample)
        double imu_ax = 0, imu_ay = 0, imu_az = 0;
        double imu_wx = 0, imu_wy = 0, imu_wz = 0;

        if (!radar_frame.imu_measurements.empty()) 
        {
            const auto& imu = radar_frame.imu_measurements.back();
            imu_ax = imu.linear_acceleration.x();
            imu_ay = imu.linear_acceleration.y();
            imu_az = imu.linear_acceleration.z();
            imu_wx = imu.angular_velocity.x();
            imu_wy = imu.angular_velocity.y();
            imu_wz = imu.angular_velocity.z();
        }

        eval_file << std::fixed << std::setprecision(6)
                << radar_frame.timestamp << ","
                << radar_frame.reve_velocity_body.linear_velocity.x() << ","
                << radar_frame.reve_velocity_body.linear_velocity.y() << ","
                << radar_frame.reve_velocity_body.linear_velocity.z() << ","
                << radar_frame.delta_yaw << ","
                << imu_ax << "," << imu_ay << "," << imu_az << ","
                << imu_wx << "," << imu_wy << "," << imu_wz << ","
                << dt_radar << "\n";
// ===========================================================

        // Add radar frame as a node in the graph
        optimizer.addFrame(radar_frame);

        // Update the node trackers
        last_valid_radar_time = current_radar_time;
        prev_cloud = static_cloud;

        // Write optimized pose to file
        gtsam::Pose3 optimized_pose = optimizer.getCurrentPose();
        outFile << last_valid_radar_time << " "
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
                      << " (Time: " << std::fixed << std::setprecision(2) << current_radar_time 
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

