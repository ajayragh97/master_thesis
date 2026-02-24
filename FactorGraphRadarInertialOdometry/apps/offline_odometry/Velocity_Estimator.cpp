#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cmath>

// NEW: Modular Includes
#include "radar_common/Types.h"
#include "radar_velocity_estimation/Estimator.h"
#include "radar_velocity_estimation/BodyFrameCorrector.h"

namespace fs = std::filesystem;

// NEW: Use specific namespaces
using namespace radar::common;
using namespace radar::velocity;

// --- Helper Functions ---

std::vector<double> loadTimestamps(const std::string& path)
{
    std::vector<double> timestamps;
    std::ifstream file(path);
    std::string line;
    if (!file.is_open()) return timestamps;
    
    while(std::getline(file, line))
    {
        if (!line.empty())
        {
            timestamps.push_back(std::stod(line));
        }
    }
    return timestamps;
}

void computePolarAndDirection(RadarTarget& target)
{
    const double x = target.position_cart.x();
    const double y = target.position_cart.y();
    const double z = target.position_cart.z();

    target.position_sph[0] = target.position_cart.norm();
    
    if (target.position_sph[0] < 0.0001) 
    {
        target.position_sph[1] = 0.0;
        target.position_sph[2] = 0.0;
        target.direction_vector = Eigen::Vector3d::Zero();
        return;
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
    if (!file.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return cloud;
    }

    // determine number of points from file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = file_size / sizeof(float); // data saved as float32 
    size_t num_points = num_floats / 5; // each row x, y, z, i, v

    std::vector<float> buffer(num_floats);
    if (file.read(reinterpret_cast<char*>(buffer.data()), file_size))
    {
        cloud.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i)
        {
            RadarTarget t;
            size_t base = i * 5;
            t.position_cart = Eigen::Vector3d(buffer[base + 0], buffer[base + 1], buffer[base + 2]);
            t.snr_db = static_cast<double>(buffer[base + 3]);
            t.doppler_velocity = static_cast<double>(buffer[base + 4]);

            computePolarAndDirection(t);
            cloud.push_back(t);
        }
    }
    return cloud;
}

// --- Main ---

int main()
{
    std::string base_dir = "/home/ajay/work/temp/aspen_run7/";
    std::string data_dir = base_dir +  "single_chip/pointclouds/data/";
    std::string stamp_path = base_dir + "single_chip/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    
    // Output
    std::string output_path = base_dir + "reve/single_chip_velocity_estimate_modular.txt";
    
    // Calibration
    std::string calib_dir = "/home/ajay/work/temp/calib/";
    std::string tf_single_chip_path = calib_dir + "transforms/base_to_single_chip.txt";
    std::string tf_imu_path = calib_dir + "transforms/base_to_imu.txt";

    // --- INITIALIZATION ---
    EstimatorConfig config;
    config.ransac_iter = 200;
    config.inlier_thresh = 0.15;
    config.min_db = 55.0;

    // Renamed variables to avoid shadowing class names
    RadarEgoEstimator estimator(config);
    BodyFrameCorrector corrector;

    // --- LOAD TRANSFORMS ---
    Transform t_radar, t_imu;
    bool tf_ok = true;
    tf_ok &= corrector.loadTransform(tf_single_chip_path, t_radar);
    tf_ok &= corrector.loadTransform(tf_imu_path, t_imu);

    if (!tf_ok)
    {
        std::cerr << "[Error] Failed to load transforms. Check paths." << std::endl;
        return -1;
    }

    corrector.setTransform(t_radar, t_imu);

    // --- LOAD IMU ---
    if (!corrector.loadImuData(imu_data_path, imu_stamp_path))
    {
        std::cerr << "[Error] Failed to load IMU data." << std::endl;
        return -1;
    }

    // --- SETUP OUTPUT ---
    std::ofstream outFile(output_path);
    if (!outFile.is_open())
    {
        std::cerr << "[Error] Could not open output file: " << output_path << std::endl;
        return -1;
    }

    // --- LOAD TIMESTAMPS ---
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    if(timestamps.empty())
    {
        std::cerr << "[Error] No timestamps found in: " << stamp_path << std::endl;
        return -1;
    }

    std::cout << "Loaded " << timestamps.size() << " frames." << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Frame" << " | " 
              << std::setw(15) << "Timestamp" << " | " 
              << std::setw(25) << "Ego-Vel (Body Frame)" << " | " 
              << std::setw(10) << "Inlier %" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // --- PROCESSING LOOP ---
    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";

        if (!fs::exists(bin_path)) continue;

        PointCloud cloud = loadPointCloud(bin_path);
        if (cloud.empty()) continue;

        // 1. ESTIMATE (Sensor Frame)
        VelocityEstimate res = estimator.estimate(cloud, false);

        if (res.success)
        {
            // 2. CORRECT (Body Frame)
            Eigen::Vector3d v_body = corrector.correctVelocity(res.linear_velocity, timestamps[i]);

            std::cout << std::setw(8) << i << " | " 
                      << std::setw(15) << timestamps[i] << " | " 
                      << std::setw(6) << v_body.x() << ", " 
                      << std::setw(6) << v_body.y() << ", " 
                      << std::setw(6) << v_body.z() << " | "
                      << std::setw(8) << res.inlier_ratio  << "%" << std::endl;

            outFile << std::fixed << std::setprecision(6) 
                    << timestamps[i] << " "
                    << v_body.x() << " "
                    << v_body.y() << " "
                    << v_body.z() << "\n";
        }
        else
        {
            std::cout << std::setw(8) << i << " | " 
                      << std::setw(15) << timestamps[i] << " | " 
                      << "       Estimation Failed      | " << std::endl;
        }
    }

    outFile.close();
    std::cout << "Processing Complete. Data saved to: " << output_path << std::endl;
    return 0;
}