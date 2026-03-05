#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>  

namespace radar
{
    namespace common
    {
        void computePolarAndDirection(RadarTarget& target);
        PointCloud loadPointCloud(const std::string& filename);
        std::vector<double> loadTimestamps(const std::string& path);
        PointCloud filterDynamicPoints(const PointCloud& input, const Eigen::Vector3d& ego_vel, double threshold = 0.5);
        std::vector<ImuData> loadImuDataVector(const std::string& data_path, const std::string& time_path);
    }
}