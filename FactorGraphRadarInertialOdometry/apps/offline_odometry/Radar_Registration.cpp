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

namespace fs = std::filesystem;
using namespace radar::common;
using namespace radar::velocity;
using namespace radar::registration;

// --- Helper Functions (Same as before) ---
void computePolarAndDirection(RadarTarget& target) {
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

PointCloud loadPointCloud(const std::string& filename) {
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

std::vector<double> loadTimestamps(const std::string& path) {
    std::vector<double> timestamps;
    std::ifstream file(path); std::string line;
    while(std::getline(file, line)) if (!line.empty()) timestamps.push_back(std::stod(line));
    return timestamps;
}

PointCloud filterDynamicPoints(const PointCloud& input, const Eigen::Vector3d& ego_vel, double threshold = 0.5) {
    PointCloud static_cloud; static_cloud.reserve(input.size());
    for(const auto& p : input) {
        double expected = -ego_vel.dot(p.direction_vector);
        if(std::abs(p.doppler_velocity - expected) < threshold) static_cloud.push_back(p);
    }
    return static_cloud;
}

int main() {
    // --- PATHS ---
    std::string base_dir = "/home/ajay/work/temp/aspen_run7/";
    std::string data_dir = base_dir +  "single_chip/pointclouds/data/";
    std::string stamp_path = base_dir + "single_chip/pointclouds/timestamps.txt";
    // This output file will be in TUM format (compatible with GT)
    std::string output_path = base_dir + "reve/single_chip_estimated_trajectory_tum.txt";

    // Calibration Paths
    std::string calib_dir = "/home/ajay/work/temp/calib/";
    std::string tf_radar_path = calib_dir + "transforms/base_to_single_chip.txt";
    std::string tf_imu_path = calib_dir + "transforms/base_to_imu.txt"; 
    bool perform_2d_icp = true;

    // --- INIT MODULES ---
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
    if(!corrector.loadTransform(tf_radar_path, t_dummy)) {
        std::cerr << "Failed to load radar transform!" << std::endl; return -1;
    }
   
    corrector.setTransform(t_dummy, t_dummy); 

    // Get T_base_radar (Static Transform)
    Eigen::Matrix4d T_base_radar = corrector.getRadarToBaseTransform();
    Eigen::Matrix4d T_radar_base = T_base_radar.inverse();

    // --- VARIABLES ---
    Eigen::Matrix4d T_global_body = Eigen::Matrix4d::Identity(); // Robot Pose (Body Frame)
    Eigen::Vector3d start_t(icp_config.start_tx, icp_config.start_ty, icp_config.start_tz); 
    double yaw_start = atan2(2.0*(icp_config.start_qw * icp_config.start_qz + icp_config.start_qx * icp_config.start_qy), 
                         1.0 - 2.0*(icp_config.start_qy * icp_config.start_qy + icp_config.start_qz * icp_config.start_qz));
    
    T_global_body.block<3,3>(0,0) = Eigen::AngleAxisd(yaw_start, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_global_body.block<3,1>(0,3) = start_t;
    std::cout << "Starting pose initialized to :\n" << T_global_body << std::endl;

    PointCloud prev_cloud;
    bool is_first_frame = true;

    // Load Timestamps
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    if(timestamps.empty()) { std::cerr << "No timestamps!" << std::endl; return -1; }

    std::ofstream outFile(output_path);
    outFile << std::fixed << std::setprecision(6);

    std::cout << "Starting Processing " << timestamps.size() << " frames..." << std::endl;

    for (size_t i = 0; i < timestamps.size(); ++i) 
    {
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";
        if (!fs::exists(bin_path)) continue;

        PointCloud raw_cloud = loadPointCloud(bin_path);
        if (raw_cloud.size() < 10) continue;

        // 1. Ego Velocity (Sensor Frame) for Filtering
        VelocityEstimate vel_res = vel_estimator.estimate(raw_cloud, false);
        Eigen::Vector3d ego_vel_sensor = vel_res.success ? vel_res.linear_velocity : Eigen::Vector3d::Zero();

        // 2. Filter Dynamic
        PointCloud static_cloud = filterDynamicPoints(raw_cloud, ego_vel_sensor);

        // 3. ICP
        if (!is_first_frame) {
            // Initial Guess (Constant Velocity or Doppler Seed)
            // Use Doppler velocity to predict translation: t = v * dt
            double dt = (i > 0) ? (timestamps[i] - timestamps[i-1]) : 0.1;
            Eigen::Vector3d t_guess_sensor = ego_vel_sensor * dt;

            // Guess Matrix (Sensor Frame, Backward T)
            Eigen::Matrix4d T_guess_icp = Eigen::Matrix4d::Identity();
            T_guess_icp.block<3,1>(0,3) = t_guess_sensor; 

            // Compute T_sensor(t -> t-1)
            Eigen::Matrix4d T_icp_sensor = icp_solver.compute(static_cloud, prev_cloud, T_guess_icp, perform_2d_icp);
            
            // Get Forward Sensor Motion: T_sensor(t-1 -> t)
            Eigen::Matrix4d M_sensor = T_icp_sensor;

            // Convert to Body Motion: M_body = T_b_r * M_sensor * T_r_b
            Eigen::Matrix4d M_body = T_base_radar * M_sensor * T_radar_base;

            // Restricting motion to only 2D
            double delta_yaw = atan2(M_body(1,0), M_body(0,0));
            double delta_x = M_body(0,3);
            double delta_y = M_body(1,3);

            M_body = Eigen::Matrix4d::Identity();
            M_body.block<3,3>(0,0) = Eigen::AngleAxisd(delta_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            M_body(0,3) = delta_x;
            M_body(1,3) = delta_y;
            M_body(2,3) = 0;
            

            // Accumulate
            T_global_body = T_global_body * M_body;
            
            std::cout << "Frame " << i << " | Body Pos: " << T_global_body.block<3,1>(0,3).transpose() << std::endl;
        }

        // 4. Save Pose (Body Frame)
        Eigen::Vector3d t = T_global_body.block<3,1>(0,3);
        Eigen::Quaterniond q(T_global_body.block<3,3>(0,0));
        q.normalize();

        // TUM Format: timestamp x y z qx qy qz qw
        outFile << timestamps[i] << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

        prev_cloud = static_cloud;
        is_first_frame = false;
    }

    std::cout << "Trajectory saved to: " << output_path << std::endl;
    return 0;
}