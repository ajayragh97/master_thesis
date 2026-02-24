#include <iostream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "Estimator.h"
#include "BodyFrameCorrector.h"

namespace fs = std::filesystem;
using namespace radar_estimator;

PointCloud generateTestData(Eigen::Vector3d true_ego_vel, int num_points, Eigen::Vector3d outlier_vel) {
    PointCloud cloud;
    
    // 1. Generate stationary background points (e.g., walls, trees)
    for (int i = 0; i < num_points; ++i) {
        RadarTarget t;
        // Random position in front of the radar (x: 1-20m, y: -10 to 10m, z: -1 to 1m)
        t.position_cart = Eigen::Vector3d((rand() % 200) / 10.0 + 1.0, 
                                     (rand() % 200 - 100) / 10.0, 
                                     (rand() % 20 - 10) / 10.0);
        
        // Doppler math: v_d = - (v_ego dot unit_direction)
        t.position_sph[0] = t.position_cart.norm();
        t.position_sph[1] = std::atan2(t.position_cart[1], t.position_cart[0]);
        double horizontalDist = t.position_cart.head<2>().norm();
        t.position_sph[2] = std::atan2(t.position_cart[2], horizontalDist);
        t.direction_vector = t.position_cart / t.position_sph[0];

        t.doppler_velocity = -true_ego_vel.dot(t.direction_vector);
        // Add a tiny bit of sensor noise (± 0.05 m/s)
        t.doppler_velocity += (rand() % 100 - 70) / 1000.0;
        
        t.snr_db = 80.0; // Strong signal
        cloud.push_back(t);
    }

    // 2. Generate a "Moving Car" (Outliers)
    // These points will follow a different velocity, which should be rejected by RANSAC
    for (int i = 0; i < 5; ++i) {
        RadarTarget t;
        t.position_cart = Eigen::Vector3d(10.0, 2.0, 0.0); // A car 10m away
        t.position_sph[0] = t.position_cart.norm();
        t.position_sph[1] = std::atan2(t.position_cart[1], t.position_cart[0]);
        double horizontalDist = t.position_cart.head<2>().norm();
        t.position_sph[2] = std::atan2(t.position_cart[2], horizontalDist);
        t.direction_vector = t.position_cart / t.position_sph[0];
        
        // This car is moving independently
        t.doppler_velocity = -outlier_vel.dot(t.direction_vector); 
        t.snr_db = 60.0;
        cloud.push_back(t);
    }

    return cloud;
}

std::vector<double> loadTimestamps(const std::string& path)
{
    std::vector<double> timestamps;
    std::ifstream file(path);
    std::string line;
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
    
    if (target.position_sph[0] < 0.0001) // prevent division by zero
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

int main()
{
    // data path
    std::string base_dir = "/media/ajay/Backup_Plus1/datasets/coloradar/kitti/2_28_2021_outdoors_run7/";
    std::string data_dir = base_dir +  "single_chip/pointclouds/data/";
    std::string stamp_path = base_dir + "single_chip/pointclouds/timestamps.txt";
    std::string imu_data_path = base_dir + "imu/imu_data.txt";
    std::string imu_stamp_path = base_dir + "imu/timestamps.txt";
    std::string output_path = base_dir + "reve/single_chip_velocity_estimate.txt";
    std::string calib_dir = "/home/ajay/work/temp/calib/";
    std::string tf_cascade_path = calib_dir + "transforms/base_to_cascade.txt";
    std::string tf_single_chip_path = calib_dir + "transforms/base_to_single_chip.txt";
    std::string tf_imu_path = calib_dir + "transforms/base_to_imu.txt";


    // estimator config
    EstimatorConfig config;
    config.ransac_iter = 200;
    config.inlier_thresh = 0.15;
    config.min_db = 55.0;

    RadarEgoEstimator RadarEgoEstimator(config);
    BodyFrameCorrector BodyFrameCorrector;

    Transform t_radar, t_imu;
    bool tf_ok = true;
    tf_ok &= BodyFrameCorrector.loadTransform(tf_single_chip_path, t_radar);
    tf_ok &= BodyFrameCorrector.loadTransform(tf_imu_path, t_imu);

    if (!tf_ok)
    {
        std::cerr << "failed to load transforms. Check Path..." << std::endl;
        return -1;
    }

    BodyFrameCorrector.setTransform(t_radar, t_imu);

    if (!BodyFrameCorrector.loadImuData(imu_data_path, imu_stamp_path))
    {
        std::cerr << "Failed to load IMU data" << std::endl;
        return -1;
    }

    std::ofstream outFile(output_path);
    if (!outFile.is_open())
    {
        std::cerr << "could not open output file at: " << output_path << std::endl;
        return -1;
    }

    // load timestamp
    std::vector<double> timestamps = loadTimestamps(stamp_path);
    if(timestamps.empty())
    {
        std::cerr << "No timestamps found in path: " << stamp_path << std::endl;
        return -1;
    }

    std::cout << "Loaded " << timestamps.size() << " frame timestamps." << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(10) << "Frame" << " | " 
              << std::setw(15) << "Timestamp" << " | " 
              << std::setw(25) << "Ego-Velocity (x, y, z)" << " | " 
              << std::setw(10) << "Inlier %" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        std::string bin_path = data_dir + "radar_pointcloud_" + std::to_string(i) + ".bin";

        if (!fs::exists(bin_path)) continue;

        PointCloud cloud = loadPointCloud(bin_path);
        if (cloud.empty()) continue;

        VelocityEstimate res = RadarEgoEstimator.estimate(cloud);

        if (res.success)
        {

            // transform velocity from sensor frame to body frame
            Eigen::Vector3d v_body = BodyFrameCorrector.correctVelocity(res.linear_velocity, timestamps[i]);

            std::cout << std::setw(10) << i << " | " 
                      << std::setw(15) << timestamps[i] << " | " 
                      << std::setw(4) << "(" << v_body.x() << ", " 
                      << std::setw(4) << v_body.y() << ", " 
                      << std::setw(4) << v_body.z() << ") | "
                      << std::setw(8) << res.inlier_ratio  << "%" << std::endl;

            outFile << std::fixed << std::setprecision(6) 
                    << timestamps[i] << " "
                    << v_body.x() << " "
                    << v_body.y() << " "
                    << v_body.z() << "\n";
        }

        else
        {
            std::cout << std::setw(10) << i << " | " 
                      << std::setw(15) << timestamps[i] << " | " 
                      << "Estimation Failed (Too few points)" << std::endl;
        }
    }

    outFile.close();
    std::cout << "Output saved to : " << output_path << std::endl;
    return 0;
}

// Test code with generated data
// int main()
// {
//     EstimatorConfig config;
//     config.ransac_iter = 200;
//     config.inlier_thresh = 0.2;

//     RadarEgoEstimator RadarEgoEstimator(config);

//     std::cout << std::fixed << std::setprecision(3);
//     std::cout << "==========================================" << std::endl;
//     std::cout << "   RADAR EGO-VELOCITY ESTIMATOR TEST      " << std::endl;
//     std::cout << "==========================================" << std::endl;

//     // --- TEST 1: Moving Forward ---
//     Eigen::Vector3d true_vel(5.0, 0.0, 0.0); // Moving 5m/s on X-axis
//     Eigen::Vector3d car_vel(15.0, 0.0, 0.0); // Another car moving faster
    
//     PointCloud moving_data = generateTestData(true_vel, 10000, car_vel);
//     VelocityEstimate res = RadarEgoEstimator.estimate(moving_data);

//     if (res.success) {
//         std::cout << "[Test 1: Moving Forward]" << std::endl;
//         std::cout << "  Simulated Vel: 5.000, 0.000, 0.000" << std::endl;
//         std::cout << "  Estimated Vel: " << res.linear_velocity.transpose() << std::endl;
//         std::cout << "  Inlier Ratio:  " << res.inlier_ratio  << "%" << std::endl;
//     } else {
//         std::cout << "[Test 1] Failed to estimate velocity!" << std::endl;
//     }

//     std::cout << "------------------------------------------" << std::endl;

//     // --- TEST 2: Stationary ---
//     Eigen::Vector3d zero_vel(0.0, 0.0, 0.0);
//     PointCloud stationary_data = generateTestData(zero_vel, 100000, car_vel);
//     res = RadarEgoEstimator.estimate(stationary_data);

//     if (res.success) {
//         std::cout << "[Test 2: Stationary]" << std::endl;
//         std::cout << "  Simulated Vel: 0.000, 0.000, 0.000" << std::endl;
//         std::cout << "  Estimated Vel: " << res.linear_velocity.transpose() << std::endl;
//     }

//     std::cout << "==========================================" << std::endl;

//     return 0;
// }