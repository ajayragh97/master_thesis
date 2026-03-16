#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

#include "radar_common/Types.h"
#include "radar_common/SystemConfig.h"
#include "radar_velocity_estimation/Estimator.h"
#include "radar_velocity_estimation/BodyFrameCorrector.h"
#include "radar_registration/RadarICP.h"
#include "factor_graph_optimization/GraphOptimizer.h"

namespace fs = std::filesystem;
using namespace radar::common;
using namespace radar::velocity;
using namespace radar::registration;
using namespace radar::optimization;

void computePolarAndDirection(RadarTarget& target) 
{
    const double x = target.position_cart.x();
    const double y = target.position_cart.y();
    const double z = target.position_cart.z();
    target.position_sph[0] = target.position_cart.norm();
    if (target.position_sph[0] < 0.0001) {
        target.position_sph[1] = 0.0; target.position_sph[2] = 0.0;
        target.direction_vector = Eigen::Vector3d::Zero(); return;
    }
    target.direction_vector = target.position_cart / target.position_sph[0];
    target.position_sph[1] = std::atan2(y, x);
    double horizontal_dist = std::sqrt(x*x + y*y);
    target.position_sph[2] = std::atan2(z, horizontal_dist);
}

PointCloud loadPointCloud(const std::string& filename) 
{
    PointCloud cloud;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return cloud;
    file.seekg(0, std::ios::end); size_t file_size = file.tellg(); file.seekg(0, std::ios::beg);
    size_t num_floats = file_size / sizeof(float); size_t num_points = num_floats / 5;
    std::vector<float> buffer(num_floats);
    if (file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        cloud.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            RadarTarget t; size_t base = i * 5;
            t.position_cart = Eigen::Vector3d(buffer[base + 0], buffer[base + 1], buffer[base + 2]);
            t.snr_db = static_cast<double>(buffer[base + 3]);
            t.doppler_velocity = static_cast<double>(buffer[base + 4]);
            computePolarAndDirection(t); cloud.push_back(t);
        }
    }
    return cloud;
}

std::vector<double> loadTimestamps(const std::string& path) 
{
    std::vector<double> timestamps;
    std::ifstream file(path); std::string line;
    while(std::getline(file, line)) if (!line.empty()) timestamps.push_back(std::stod(line));
    return timestamps;
}

PointCloud filterDynamicPoints(const PointCloud& input, const Eigen::Vector3d& ego_vel, double threshold = 0.5) 
{
    PointCloud static_cloud; static_cloud.reserve(input.size());
    for(const auto& p : input) {
        double expected = -ego_vel.dot(p.direction_vector);
        if(std::abs(p.doppler_velocity - expected) < threshold) static_cloud.push_back(p);
    }
    return static_cloud;
}

std::vector<ImuData> loadImuDataVector(const std::string& data_path, const std::string& time_path)
{
    std::vector<ImuData> imu_vec;
    std::ifstream f_data(data_path);
    std::ifstream f_time(time_path);

    if (!f_data.is_open() || !f_time.is_open()) return imu_vec;

    double t, ax, ay, az, gx, gy, gz;
    while (f_time >> t)
    {
        if (!(f_data >> ax >> ay >> az >> gx >> gy >> gz)) break;
        
        ImuData data;
        data.timestamp = t;
        data.linear_acceleration = Eigen::Vector3d(ax, ay, az);
        data.angular_velocity = Eigen::Vector3d(gx, gy, gz);
        imu_vec.push_back(data);
    }
    return imu_vec;
}

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

    PointCloud prev_cloud;
    size_t current_imu_idx = 0; // To keep track of IMU data 
    while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp < timestamps[start_idx]) 
    {
        current_imu_idx++;
    }
    std::cout << "Starting Factor Graph Processing..." << std::endl;
    
    // Main Loop
    for (size_t i = 0; i < timestamps.size(); ++i)
    {

        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";
        if (!fs::exists(bin_path)) continue;

        PointCloud raw_cloud = loadPointCloud(bin_path);
        if (raw_cloud.size() < 10) continue;

        // 1. Estimate Ego Velocity using REVE (Sensor Frame)
        VelocityEstimate vel_estimate_sensor = vel_estimator.estimate(raw_cloud, false);

        // 2. Filter Dynamic Points
        PointCloud static_cloud;
        if (vel_estimate_sensor.success) 
        {
            static_cloud = filterDynamicPoints(raw_cloud, vel_estimate_sensor.linear_velocity, cfg.icp.dynamic_point_vel_thresh);
        } 
        else 
        {
            static_cloud = raw_cloud;
        }

        if (i == start_idx) 
        {
            // Do NOT add to graph. Just initialize prev_cloud for the next step's ICP.
            prev_cloud = static_cloud;
            continue; // Skip to next iteration to start graph optimization from the next frame after initialization
        }
        // ==========================================

        OptimizationFrame current_frame;
        current_frame.timestamp = timestamps[i];

        // 3. Get IMU slice between previous and current timestamp
        double prev_time = timestamps[i-1];
        while (current_imu_idx < all_imu_data.size() && all_imu_data[current_imu_idx].timestamp <= timestamps[i])
        {
            if (all_imu_data[current_imu_idx].timestamp > prev_time)
            {
                current_frame.imu_measurements.push_back(all_imu_data[current_imu_idx]);
            }
            current_imu_idx++;
        }

        // Safety check to ensure we have IMU data for this frame
        if (current_frame.imu_measurements.empty())
        {
            std::cerr << "WARNING: No IMU measurements found for frame at timestamp " << timestamps[i] << ". Skipping IMU integration for this frame." << std::endl;
        }   
        if (timestamps[i] - timestamps[i-1] <= 0.0)
        {
            std::cerr << "WARNING: Duplicate frame or non-positive time difference between frames at index " << i-1 << " and " << i << ". Skipping this frame." << std::endl;
            continue; // Skip this frame entirely
        }

        // 4. Populate REVE velocity
        if (vel_estimate_sensor.success)
        {
            Eigen::Vector3d vel_estimate_body = corrector.correctVelocity(vel_estimate_sensor.linear_velocity, timestamps[i]);
            current_frame.reve_velocity_body.linear_velocity = vel_estimate_body;
            current_frame.reve_velocity_body.inlier_ratio = vel_estimate_sensor.inlier_ratio;
            current_frame.reve_velocity_body.covariance = vel_estimate_sensor.covariance;
            current_frame.has_reve_velocity = true;
        }
        else
        {
            current_frame.has_reve_velocity = false;
        }

        // 5. Estimate delta yaw using ICP
        double dt = timestamps[i] - timestamps[i-1];
        
        // Negative velocity for ICP translation guess (Moving current cloud backwards to match prev cloud)
        Eigen::Vector3d T_guess_sensor = -vel_estimate_sensor.linear_velocity * dt; 
        Eigen::Matrix4d T_guess_icp = Eigen::Matrix4d::Identity();
        T_guess_icp.block<3,1>(0,3) = T_guess_sensor; 

        // Compute ICP
        Eigen::Matrix4d T_icp_sensor = icp_solver.compute(static_cloud, prev_cloud, T_guess_icp, cfg.graph.use_2d_mode);
        
        // Convert to Body Frame
        Eigen::Matrix4d M_sensor = T_icp_sensor;
        Eigen::Matrix4d M_body = T_base_radar * M_sensor * T_radar_base;

        // Extract delta yaw
        double delta_yaw = atan2(M_body(1,0), M_body(0,0));
        current_frame.delta_yaw = delta_yaw;
        current_frame.delta_yaw_fitness = icp_solver.getFitnessScore();
        current_frame.has_delta_yaw = true;

        // 6. Add frame to factor graph optimizer
        optimizer.addFrame(current_frame);
        
        // Progress output
        if (i % 100 == 0) {
            std::cout << "Processing frame " << i << " / " << timestamps.size() << "\r" << std::flush;
        }

        // Update previous cloud for next ICP
        prev_cloud = static_cloud;
    }

    optimizer.saveFullTrajectory(output_path);
    std::cout << "Factor Graph Optimization Completed. Trajectory saved to: " << output_path << std::endl;
    return 0;
}