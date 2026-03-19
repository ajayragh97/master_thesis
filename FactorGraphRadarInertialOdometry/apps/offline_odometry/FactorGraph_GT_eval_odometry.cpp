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
    std::string config_path = "../config.json";
    if (argc > 1) config_path = argv[1];

    SystemConfig cfg = SystemConfig::load(config_path);

    int GT_SKIP_RATE = cfg.graph.gt_skip_rate;
    std::cout << "GT skip rate: " << GT_SKIP_RATE << std::endl;
    std::string base_dir = cfg.dataset.base_dir;
    std::string data_dir = base_dir + cfg.dataset.sensor + "/pointclouds/data/";
    std::string radar_stamp_path = base_dir + cfg.dataset.sensor + "/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    std::string gt_pose_path = base_dir + "groundtruth/groundtruth_poses.txt";
    std::string gt_stamp_path = base_dir + "groundtruth/timestamps.txt";

    std::vector<double> radar_times = loadTimestamps(radar_stamp_path);
    std::vector<ImuData> all_imu_data = loadImuDataVector(imu_data_path, imu_stamp_path);
    std::vector<GTData> gt_data = loadGroundTruth(gt_pose_path, gt_stamp_path);

    BodyFrameCorrector corrector;
    Transform t_radar, t_imu; 
    corrector.loadTransform(cfg.tf_radar_path, t_radar);
    corrector.loadTransform(cfg.tf_imu_path, t_imu);
    corrector.setTransform(t_radar, t_imu);
    corrector.loadImuData(imu_data_path, imu_stamp_path);
    cfg.graph.T_base_imu = t_imu;

    Eigen::Matrix4d T_base_radar = corrector.getRadarToBaseTransform();
    Eigen::Matrix4d T_radar_base = T_base_radar.inverse();

    // --- Static Initialization Phase ---
    std::cout << "Calibrating IMU Biases & Gravity from static period..." << std::endl;
    ImuBias imu_bias = getImuBias(all_imu_data, cfg.dataset.static_duration);

    // get mean gravity from initial static duration
    double estimated_gravity = imu_bias.bias_acc.norm();

    // Bias initialization
    gtsam::imuBias::ConstantBias init_bias(gtsam::Vector3::Zero(), imu_bias.bias_gyro);


    // Velocity initialization
    gtsam::Vector3 init_velocity = gtsam::Vector3::Zero();

    // --- Phase 1: Preprocessing ---
    std::cout << "--- Starting Preprocessing ---" << std::endl;
    double poserate_dt = 1.0 / cfg.graph.keyframe_rate;
    double current_time = radar_times.front();
    double end_time = radar_times.back();

    std::vector<OptimizationFrame> synced_frames;
    size_t imu_idx = 0;
    int node_count = 0;

    RadarEgoEstimator vel_estimator(cfg.reve);
    RadarICP icp_solver(cfg.icp);
    PointCloud prev_cloud;
    double last_radar_time = current_time;

    while (current_time <= end_time) 
    {
        OptimizationFrame frame;
        frame.timestamp = current_time;

        while (imu_idx < all_imu_data.size() && all_imu_data[imu_idx].timestamp <= current_time) 
        {
            frame.imu_measurements.push_back(all_imu_data[imu_idx]);
            imu_idx++;
        }
        // ==============================================================
        // ZERO-ORDER HOLD FOR IMU GAPS
        // If an interval has no IMU data (hardware jitter), we duplicate 
        // the last known IMU measurement to ensure the Preintegrator 
        // perfectly bridges the time gap!
        // ==============================================================
        if (frame.imu_measurements.empty() && imu_idx > 0) 
        {
            ImuData synthetic_imu = all_imu_data[imu_idx - 1];
            synthetic_imu.timestamp = current_time; // Assign the current frame's time
            frame.imu_measurements.push_back(synthetic_imu);
        }

        bool is_static = (current_time - radar_times.front()) <= cfg.dataset.static_duration;
        if (is_static || (node_count % GT_SKIP_RATE == 0)) 
        {
            frame.gt_pose = getClosestGTPose(gt_data, current_time);
            frame.has_gt_pose = true;
        }

        int r_idx = getClosestRadarIdx(radar_times, current_time);
        if (std::abs(radar_times[r_idx] - current_time) < (poserate_dt / 2.0)) 
        {
            std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(r_idx) + ".bin";
            if (fs::exists(bin_path)) 
            {
                PointCloud raw_cloud = loadPointCloud(bin_path);
                if (raw_cloud.size() >= 10) 
                {
                    VelocityEstimate vel_est = vel_estimator.estimate(raw_cloud, false);
                    PointCloud static_cloud = raw_cloud;

                    if (vel_est.success) 
                    {
                        frame.reve_velocity_body = vel_est;
                        corrector.correctVelocity(frame.reve_velocity_body, current_time);
                        frame.has_reve_velocity = true;
                        static_cloud = filterDynamicPoints(raw_cloud, vel_est.linear_velocity, cfg.icp.dynamic_point_vel_thresh);
                    }

                    if (!prev_cloud.empty()) 
                    {
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

        if (!frame.imu_measurements.empty() || frame.has_gt_pose) synced_frames.push_back(frame);
        current_time += poserate_dt;
        node_count++;
    }

    gtsam::Pose3 init_pose = synced_frames[0].gt_pose;

    // --- Phase 2: Factor Graph Optimization ---
    std::cout << "--- Starting Optimization ---" << std::endl;
    GraphOptimizer optimizer(cfg.graph);
    optimizer.initialize(init_pose, init_velocity, init_bias, synced_frames[0].timestamp, estimated_gravity, imu_bias);
    std::cout << "Init IMu readings: "  << synced_frames[0].imu_measurements[0].linear_acceleration.x() << "," 
                                        << synced_frames[0].imu_measurements[0].linear_acceleration.y() << ","
                                        << synced_frames[0].imu_measurements[0].linear_acceleration.z() << ","
                                        << synced_frames[0].imu_measurements[0].angular_velocity.x() << ","
                                        << synced_frames[0].imu_measurements[0].angular_velocity.y() << ","
                                        << synced_frames[0].imu_measurements[0].angular_velocity.z() << std::endl;

    for (size_t i = 1; i < synced_frames.size(); ++i) 
    {
        optimizer.addFrame(synced_frames[i]);
        if (i % 100 == 0) std::cout << "Optimized Frame " << i << " / " << synced_frames.size() << "\r" << std::flush;
    }

    std::string out_path = base_dir + cfg.dataset.output_filename + "_final_traj.txt";
    optimizer.saveFullTrajectory(out_path);
    return 0;
}