#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>

#include "radar_common/Types.h"
#include "radar_velocity_estimation/Estimator.h"
#include "radar_registration/RadarICP.h"

namespace fs = std::filesystem;
using namespace radar::common;
using namespace radar::velocity;
using namespace radar::registration;

// --- Helper: Load Point Cloud ---
// (Copying this helper for the test script to keep it standalone)
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

// --- Helper: Save Cloud to CSV for Python ---
void saveCloudForPlotting(const std::string& filename, const PointCloud& cloud) {
    std::ofstream file(filename);
    // Format: x y z intensity
    for (const auto& p : cloud) {
        file << p.position_cart.x() << " " 
             << p.position_cart.y() << " " 
             << p.position_cart.z() << " "
             << p.snr_db << "\n";
    }
    file.close();
    std::cout << "Saved " << cloud.size() << " points to " << filename << std::endl;
}

// --- Helper: Apply Transform to Cloud ---
PointCloud transformCloud(const PointCloud& in, const Eigen::Matrix4d& T) {
    PointCloud out = in;
    for (auto& p : out) {
        Eigen::Vector4d vec(p.position_cart.x(), p.position_cart.y(), p.position_cart.z(), 1.0);
        p.position_cart = (T * vec).head<3>();
    }
    return out;
}

int main(int argc, char** argv) {
    // 1. Setup Indices (Change these to test different pairs)
    int idx_target = 171; // Previous Frame
    int idx_source = 172; // Current Frame (The one we move)

    std::string base_dir = "/home/ajay/work/temp/aspen_run7/";
    std::string data_dir = base_dir +  "cascade/pointclouds/data/";

    std::cout << "--- Radar ICP Single Frame Test ---" << std::endl;
    std::cout << "Target Frame: " << idx_target << "\nSource Frame: " << idx_source << std::endl;

    // 2. Load Clouds
    PointCloud target_cloud = loadPointCloud(data_dir + "radar_pointcloud_" + std::to_string(idx_target) + ".bin");
    PointCloud source_cloud = loadPointCloud(data_dir + "radar_pointcloud_" + std::to_string(idx_source) + ".bin");

    if (target_cloud.empty() || source_cloud.empty()) {
        std::cerr << "Error loading clouds." << std::endl; return -1;
    }

    // 3. Filter Dynamic Points (Optional but recommended)
    EstimatorConfig vel_config; vel_config.min_db = 50.0;
    RadarEgoEstimator estimator(vel_config);
    
    // Simple lambda to filter
    auto filter = [&](const PointCloud& c) {
        auto res = estimator.estimate(c, true);
        if(!res.success) return c; // Return raw if estimation fails
        PointCloud out;
        for(const auto& p : c) {
            double expected = -res.linear_velocity.dot(p.direction_vector);
            if(std::abs(p.doppler_velocity - expected) < 0.5) out.push_back(p);
        }
        return out;
    };

    PointCloud target_static = filter(target_cloud);
    PointCloud source_static = filter(source_cloud);
    std::cout << "Filtered Dynamic Points. Source: " << source_cloud.size() << " -> " << source_static.size() << std::endl;

    // 4. Run ICP
    ICPConfig icp_config;
    icp_config.max_iterations = 50;
    icp_config.max_correspondences_dist = 2.0;
    icp_config.intensity_threshold = 0.25;
    
    RadarICP icp(icp_config);
    
    // Initial Guess (Identity for now, or use small forward translation)
    Eigen::Matrix4d guess = Eigen::Matrix4d::Identity();
    
    // COMPUTE
    Eigen::Matrix4d T_result = icp.compute(source_static, target_static, guess);

    // 5. Output Results
    std::cout << "Converged Fitness Score: " << icp.getFitnessScore() << std::endl;
    std::cout << "Resulting Transform:\n" << T_result << std::endl;

    // 6. Save Files for Python Visualization
    saveCloudForPlotting("/home/ajay/work/temp/clouds/debug_target.txt", target_static);
    saveCloudForPlotting("/home/ajay/work/temp/clouds/debug_source_raw.txt", source_static);
    
    PointCloud source_aligned = transformCloud(source_static, T_result);
    saveCloudForPlotting("/home/ajay/work/temp/clouds/debug_source_aligned.txt", source_aligned);

    return 0;
}