#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

#include "radar_common/Types.h"
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

    if (!f_data.is_open()) || (!f_time.is_open()) return imu_vec;

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

int main()
{
    // Paths
    std::string base_dir = "/home/ajay/work/temp/aspen_run7/";
    std::string data_dir = base_dir +  "single_chip/pointclouds/data/";
    std::string stamp_path = base_dir + "single_chip/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    std::string output_path = base_dir + "reve/factor_graph_trajectory_single_chip.txt";

    std::string calib_dir = "/home/ajay/work/temp/calib/";
    std::string tf_radar_path = calib_dir + "transforms/base_to_single_chip.txt";
    std::string tf_imu_path = calib_dir + "transforms/base_to_imu.txt"; 

    // Init 
    EstimatorConfig vel_config; 
    vel_config.ransac_iter = 150; 
    vel_config.inlier_thresh = 0.2; 
    vel_config.min_db = 55.0;
    RadarEgoEstimator vel_estimator(vel_config);

    ICPConfig icp_config; 
    icp_config.max_iterations = 30; 
    icp_config.max_correspondences_dist = 1.5; 
    icp_config.intensity_threshold = 0.25;
    RadarICP icp_solver(icp_config);

    BodyFrameCorrector corrector;
    Transform t_dummy; 
    corrector.loadTransform(tf_radar_path, t_dummy);
    corrector.setTransform(t_dummy, t_dummy); 
    corrector.loadImuData(imu_data_path, imu_stamp_path); // for omega interpolation

    Eigen::Matrix4d T_base_radar = corrector.getRadarToBaseTransform();
    Eigen::Matrix4d T_radar_base = T_base_radar.inverse();

    GraphOptimizer optimizer;

    // Load data
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    std::vector<ImuData> all_imu_data = loadImuDataVector(imu_data_path, imu_stamp_path);

    if (timestamps.empty() || all_imu_data.empty())
    {
        std::cerr << "Missing imu data or timestamps" << std::endl;
        return -1;
    }

    std::ofstream outFile(output_path);
    outFile << std::fixed << std::setprecision(6);

    // initialize graph
    gtsam::Pose3 init_pose(gtsam::Rot3::Quaternion(icp_config.start_qw, icp_config.start_qx, icp_config.start_qy, icp_config.start_qz),
                           gtsam::Point3(icp_config.start_tx, icp_config.start_ty, icp_config.start_tz));
    
    gtsam::Vector3 init_velocity = gtsam::Vector3::Zero();
    gtsam::imuBias::ConstantBias init_bias;

    optimizer.initialize(init_pose, init_velocity, init_bias, timestamps[0]);

    PointCloud prev_cloud;
    size_t current_imu_idx = 0; // To keep track of IMU data 
    std::cout << "Starting Factor Graph Processing..." << std::endl;
    
    // Main Loop
    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";
        if (!fs::exists(bin_path)) continue;

        PointCloud raw_cloud = loadPointCloud(bin_path);
        if (raw_cloud.size() < 10) continue;

        OptimizationFrame current_frame;
        current_frame.timestamp = timestamps[i];

        // ToDo...
    }
}