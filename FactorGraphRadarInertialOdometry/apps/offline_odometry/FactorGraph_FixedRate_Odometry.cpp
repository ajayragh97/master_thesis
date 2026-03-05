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
    std::string config_path = "config.json"; // Default config path
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
    std::string output_path = base_dir + cfg.dataset.output_filename;

    // Load data
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    std::vector<ImuData> all_imu_data = loadImuDataVector(imu_data_path, imu_stamp_path);

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
    gtsam::Pose3 init_pose(gtsam::Rot3::Quaternion(cfg.icp.start_qw, cfg.icp.start_qx, cfg.icp.start_qy, cfg.icp.start_qz),
                           gtsam::Point3(cfg.icp.start_tx, cfg.icp.start_ty, cfg.icp.start_tz));
    
    gtsam::Vector3 init_velocity = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias init_bias;

    optimizer.initialize(init_pose, init_velocity, init_bias, timestamps[start_idx]);

    // State tracking
    double last_node_timestamp = timestamps[start_idx];
    size_t current_imu_idx = 0;

    // fast forward IMU to start
    while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp < last_node_timestamp) 
    {
        current_imu_idx++;
    }

    PointCloud prev_cloud;
    std::cout << "Starting Fixed-Rate Optimization (Target: " << (1.0/cfg.graph.keyframe_rate) << "Hz)..." << std::endl;

    // Main Loop
    for (size_t i = start_idx; i < timestamps.size(); ++i)
    {
        // 1. Process radar data
    }

}

