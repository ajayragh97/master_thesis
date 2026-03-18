 /* Working Gt aided factor graph code */
 
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
#include "factor_graph_optimization/RadarVelocityFactor.h"

// GTSAM Headers
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>




namespace fs = std::filesystem;
using namespace radar::common;
using namespace radar::velocity;
using namespace radar::registration;
using namespace radar::optimization;

using gtsam::symbol_shorthand::X; // Pose
using gtsam::symbol_shorthand::V; // Velocity
using gtsam::symbol_shorthand::B; // Bias

// =========================================================================
// HELPER FUNCTIONS FOR GROUND TRUTH
// =========================================================================
// struct GTData 
// {
//     double timestamp;
//     gtsam::Pose3 pose;
// };

// std::vector<GTData> loadGroundTruth(const std::string& pose_file, const std::string& time_file) 
// {
//     std::vector<GTData> gt_list;
//     std::ifstream f_pose(pose_file), f_time(time_file);
//     if (!f_pose.is_open() || !f_time.is_open()) return gt_list;

//     double t, x, y, z, qx, qy, qz, qw;
//     while (f_time >> t && f_pose >> x >> y >> z >> qx >> qy >> qz >> qw) 
//     {
//         // Note: GTSAM Quaternion constructor is (w, x, y, z)
//         gtsam::Rot3 rot = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
//         gtsam::Point3 trans(x, y, z);
//         gt_list.push_back({t, gtsam::Pose3(rot, trans)});
//     }
//     return gt_list;
// }

// gtsam::Pose3 getClosestGTPose(const std::vector<GTData>& gt_list, double target_time) 
// {
//     auto it = std::min_element(gt_list.begin(), gt_list.end(),
//         [target_time](const GTData& a, const GTData& b) {
//             return std::abs(a.timestamp - target_time) < std::abs(b.timestamp - target_time);
//         });
//     return it->pose;
// }

// int getClosestRadarIdx(const std::vector<double>& timestamps, double target_time) 
// {
//     auto it = std::min_element(timestamps.begin(), timestamps.end(),
//         [target_time](double a, double b) {
//             return std::abs(a - target_time) < std::abs(b - target_time);
//         });
//     return std::distance(timestamps.begin(), it);
// }

// Convert Eigen 4x4 to GTSAM Pose3
gtsam::Pose3 convertEigenToPose3(const Eigen::Matrix4d& T) 
{
    gtsam::Rot3 rot(T.block<3,3>(0,0));
    gtsam::Point3 trans(T(0,3), T(1,3), T(2,3));
    return gtsam::Pose3(rot, trans);
}
// =========================================================================

int main(int argc, char** argv)
{
    std::string config_path = "../config.json";
    if (argc > 1) config_path = argv[1];

    SystemConfig cfg = SystemConfig::load(config_path);

    // Dynamic configuration variables
    int GT_SKIP_RATE = cfg.graph.gt_skip_rate; // 1 = use every frame, 50 = use 1 GT pose per 50 frames
    int ISAM2_UPDATE_RATE = cfg.graph.update_rate; // Batch optimize every 10 frames

    // Paths
    std::string base_dir = cfg.dataset.base_dir;
    std::string data_dir = base_dir + cfg.dataset.sensor + "/pointclouds/data/";
    std::string stamp_path = base_dir + cfg.dataset.sensor + "/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    std::string gt_pose_path = base_dir + "groundtruth/groundtruth_poses.txt";
    std::string gt_time_path = base_dir + "groundtruth/timestamps.txt";
    
    std::string final_output_path = base_dir + cfg.dataset.output_filename + "_final_traj.txt";

    // Load Data
    std::vector<double> radar_times = loadTimestamps(stamp_path);
    std::vector<ImuData> all_imu_data = loadImuDataVector(imu_data_path, imu_stamp_path);
    std::vector<GTData> gt_data = loadGroundTruth(gt_pose_path, gt_time_path);

    if (radar_times.empty() || all_imu_data.empty() || gt_data.empty()) 
    {
        std::cerr << "CRITICAL: Missing Sensor or GT Data!" << std::endl; return -1;
    }

    // Load Calibrations
    BodyFrameCorrector corrector;
    Transform t_radar, t_imu; 
    corrector.loadTransform(cfg.tf_radar_path, t_radar);
    corrector.loadTransform(cfg.tf_imu_path, t_imu);
    corrector.setTransform(t_radar, t_imu);
    corrector.loadImuData(imu_data_path, imu_stamp_path);
    
    Eigen::Matrix4d T_base_radar = corrector.getRadarToBaseTransform();
    Eigen::Matrix4d T_radar_base = T_base_radar.inverse();

    // Init Modules
    RadarEgoEstimator vel_estimator(cfg.reve);
    RadarICP icp_solver(cfg.icp);

    // =========================================================================
    // PHASE 1: DATA PREPROCESSING (Build the Synchronized Timeline)
    // =========================================================================
    std::cout << "\n--- Starting Phase 1: Data Preprocessing ---" << std::endl;

    double poserate_dt = 1.0 / cfg.graph.keyframe_rate; // e.g., 0.02s (50Hz)
    double current_time = radar_times.front();
    double end_time = radar_times.back();
    
    std::vector<OptimizationFrame> synced_frames;
    size_t imu_idx = 0;
    int node_count = 0;
    
    PointCloud prev_cloud;
    double last_radar_time = current_time;

    while (current_time <= end_time) 
    {
        OptimizationFrame frame;
        frame.timestamp = current_time;

        // 1. Bin IMU Measurements
        while (imu_idx < all_imu_data.size() && all_imu_data[imu_idx].timestamp <= current_time) 
        {
            frame.imu_measurements.push_back(all_imu_data[imu_idx]);
            imu_idx++;
        }

        // 2. Assign Ground Truth Pose
        bool is_static_period = (current_time - radar_times.front()) <= cfg.dataset.static_duration;
        if (is_static_period || (node_count % GT_SKIP_RATE == 0)) {
            frame.gt_pose = getClosestGTPose(gt_data, current_time);
            frame.has_gt_pose = true;
        }

        // 3. Assign Radar Data (If a scan exists near this timestamp)
        int r_idx = getClosestRadarIdx(radar_times, current_time);
        if (std::abs(radar_times[r_idx] - current_time) < (poserate_dt / 2.0)) 
        {
            std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(r_idx) + ".bin";
            if (fs::exists(bin_path)) 
            {
                PointCloud raw_cloud = loadPointCloud(bin_path);
                if (raw_cloud.size() >= 10) 
                {
                    // Evaluate REVE
                    VelocityEstimate vel_est = vel_estimator.estimate(raw_cloud, false);
                    PointCloud static_cloud = raw_cloud;

                    if (vel_est.success) {
                        frame.reve_velocity_body = vel_est;
                        frame.reve_velocity_body.linear_velocity = corrector.correctVelocity(vel_est.linear_velocity, current_time);
                        frame.has_reve_velocity = true;
                        static_cloud = filterDynamicPoints(raw_cloud, vel_est.linear_velocity, cfg.icp.dynamic_point_vel_thresh);
                    }

                    // Evaluate ICP
                    if (!prev_cloud.empty()) {
                        double dt_radar = current_time - last_radar_time;
                        Eigen::Vector3d T_guess_sensor = vel_est.linear_velocity * dt_radar;
                        Eigen::Matrix4d T_guess_icp = Eigen::Matrix4d::Identity();
                        T_guess_icp.block<3,1>(0,3) = T_guess_sensor;

                        Eigen::Matrix4d T_icp_sensor = icp_solver.compute(static_cloud, prev_cloud, T_guess_icp, false);
                        frame.T_body = T_base_radar * T_icp_sensor * T_radar_base;
                        frame.icp_fitness = icp_solver.getFitnessScore();
                        frame.has_icp = true;
                    }

                    prev_cloud = static_cloud;
                    last_radar_time = current_time;
                }
            }
        }

        // Only add frames that actually have IMU data to bridge them
        if (!frame.imu_measurements.empty()) 
        {
            synced_frames.push_back(frame);
        }
        
        current_time += poserate_dt;
        node_count++;
    }

    std::cout << "Successfully generated " << synced_frames.size() << " synchronized frames." << std::endl;

    // =========================================================================
    // PHASE 2: FACTOR GRAPH OPTIMIZATION
    // =========================================================================
    std::cout << "\n--- Starting Phase 2: Factor Graph Optimization ---" << std::endl;

    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = cfg.graph.isam2_relinearize_threshold;
    isam_params.relinearizeSkip = cfg.graph.isam2_relinearize_skip;
    gtsam::ISAM2 isam2(isam_params);

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_values;

    // Setup IMU Preintegrator
    auto imu_params = gtsam::PreintegrationParams::MakeSharedU(cfg.graph.gravity);
    imu_params->setAccelerometerCovariance(gtsam::Matrix33::Identity() * std::pow(cfg.graph.accel_noise_sigma, 2));
    imu_params->setGyroscopeCovariance(gtsam::Matrix33::Identity() * std::pow(cfg.graph.gyro_noise_sigma, 2));
    imu_params->setIntegrationCovariance(gtsam::Matrix33::Identity() * 1e-8);

    gtsam::Rot3 imu_rotation = gtsam::Rot3::Roll(M_PI); 
    gtsam::Point3 imu_translation(t_imu.translation); // Keep the physical lever-arm
    imu_params->setBodyPSensor(gtsam::Pose3(imu_rotation, imu_translation));
    
    // Initial Bias (Use static calculation if you have it, else Zero)
    gtsam::imuBias::ConstantBias current_bias; 
    auto preintegrated_imu = std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params, current_bias);

    // Initial State (Node 0)
    gtsam::Pose3 current_pose = synced_frames[0].gt_pose; // Anchor perfectly to the first GT pose
    gtsam::Vector3 current_velocity = gtsam::Vector3::Zero();

    // Node 0 Priors
    auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4).finished());
    auto vel_noise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 1e-2, 1e-2, 1e-2).finished());
    auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-3, 1e-3, 1e-3).finished());

    graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), current_pose, pose_noise));
    graph.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), current_velocity, vel_noise));
    graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), current_bias, bias_noise));

    initial_values.insert(X(0), current_pose);
    initial_values.insert(V(0), current_velocity);
    initial_values.insert(B(0), current_bias);

    isam2.update(graph, initial_values);
    graph.resize(0); initial_values.clear();

    // Main Graph Building Loop
    int update_counter = 0;
    uint64_t last_radar_node_idx = 0;

    for (size_t i = 1; i < synced_frames.size(); ++i)
    {
        const auto& frame = synced_frames[i];
        uint64_t curr_idx = i;
        uint64_t prev_idx = i - 1;

        // 1. Integrate IMU
        for (const auto& imu : frame.imu_measurements) {
            double dt = 1.0 / all_imu_data.size(); // Approximate or calculate real dt
            if (frame.imu_measurements.size() > 1) 
            {
                dt = (frame.imu_measurements.back().timestamp - frame.imu_measurements.front().timestamp) / frame.imu_measurements.size();
            }
            if (dt > 1e-6) preintegrated_imu->integrateMeasurement(imu.linear_acceleration, imu.angular_velocity, dt);
        }

        graph.add(gtsam::ImuFactor(X(prev_idx), V(prev_idx), X(curr_idx), V(curr_idx), B(prev_idx), *preintegrated_imu));

        // 2. Bias Random Walk
        double dt_frame = frame.timestamp - synced_frames[prev_idx].timestamp;
        auto rw_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
            cfg.graph.accel_bias_rw_sigma_x * std::sqrt(dt_frame), cfg.graph.accel_bias_rw_sigma_y * std::sqrt(dt_frame), cfg.graph.accel_bias_rw_sigma_z * std::sqrt(dt_frame),
            cfg.graph.gyro_bias_rw_sigma_x * std::sqrt(dt_frame), cfg.graph.gyro_bias_rw_sigma_y * std::sqrt(dt_frame), cfg.graph.gyro_bias_rw_sigma_z * std::sqrt(dt_frame)
        ).finished());
        graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(prev_idx), B(curr_idx), gtsam::imuBias::ConstantBias(), rw_noise));

        // 3. Ground Truth GPS Prior (The Anchor)
        if (frame.has_gt_pose) 
        {
            auto gt_meas_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished()); // Allow slight float
            graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(curr_idx), frame.gt_pose, gt_meas_noise));
        }

        // 4. REVE Velocity
        if (frame.has_reve_velocity) 
        {
            auto v_cov = gtsam::noiseModel::Gaussian::Covariance(frame.reve_velocity_body.covariance);
            // test dont trust reve
            // auto v_cov = gtsam::noiseModel::Isotropic::Sigma(3, 10.0); // MASSIVE NOISE
            graph.add(RadarVelocityFactor(X(curr_idx), V(curr_idx), frame.reve_velocity_body.linear_velocity, v_cov));
        }

        // 5. ICP Between Factor
        if (frame.has_icp && last_radar_node_idx != 0) 
        {
            gtsam::Pose3 relative_icp_pose = convertEigenToPose3(frame.T_body);
            double icp_sig = std::max(frame.icp_fitness, 0.01);
            
            // High confidence in Translation and Yaw, lower confidence in Roll/Pitch
            auto icp_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
                0.1, 0.1, icp_sig, icp_sig, icp_sig, icp_sig).finished());
            
            // test dont trust icp
            // auto icp_noise = gtsam::noiseModel::Isotropic::Sigma(6, 10.0); // MASSIVE NOISE
                
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_radar_node_idx), X(curr_idx), relative_icp_pose, icp_noise));
        }
        
        if (frame.has_icp) last_radar_node_idx = curr_idx;

        // Predict and Insert Initial Values
        gtsam::NavState pred_state = preintegrated_imu->predict(gtsam::NavState(current_pose, current_velocity), current_bias);
        initial_values.insert(X(curr_idx), pred_state.pose());
        initial_values.insert(V(curr_idx), pred_state.velocity());
        initial_values.insert(B(curr_idx), current_bias);

        // Update ISAM2
        update_counter++;
        if (update_counter >= ISAM2_UPDATE_RATE || i == synced_frames.size() - 1) 
        {
            isam2.update(graph, initial_values);
            graph.resize(0); initial_values.clear();

            // Reset tracking variables based on Optimized State
            gtsam::Values result = isam2.calculateEstimate();
            current_pose = result.at<gtsam::Pose3>(X(curr_idx));
            current_velocity = result.at<gtsam::Vector3>(V(curr_idx));
            current_bias = result.at<gtsam::imuBias::ConstantBias>(B(curr_idx));
            
            preintegrated_imu->resetIntegrationAndSetBias(current_bias);
            update_counter = 0;
            
            if (i % 200 == 0) std::cout << "Optimized up to frame " << i << " / " << synced_frames.size() << std::endl;
        }
    }

    // =========================================================================
    // PHASE 3: EXTRACT FINAL TRAJECTORY
    // =========================================================================
    std::cout << "\n--- Saving Final Trajectory ---" << std::endl;
    std::ofstream outFinal(final_output_path);
    outFinal << std::fixed << std::setprecision(6);

    gtsam::Values final_estimate = isam2.calculateEstimate();
    for (size_t i = 0; i < synced_frames.size(); ++i) {
        if (final_estimate.exists(X(i))) {
            gtsam::Pose3 p = final_estimate.at<gtsam::Pose3>(X(i));
            auto q = p.rotation().toQuaternion();
            gtsam::Matrix cov = isam2.marginalCovariance(X(i));
            
            outFinal << synced_frames[i].timestamp << " "
                     << p.x() << " " << p.y() << " " << p.z() << " "
                     << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                     << cov(3,3) << " " << cov(3,4) << " " << cov(3,5) << " "
                     << cov(4,4) << " " << cov(4,5) << " " << cov(5,5) << "\n";
        }
    }
    outFinal.close();
    std::cout << "Saved successfully to: " << final_output_path << std::endl;

    return 0;
}