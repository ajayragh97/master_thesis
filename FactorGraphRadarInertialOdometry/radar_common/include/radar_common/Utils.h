#pragma once

#include <string>
#include <vector>
#include <numeric>
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
        std::vector<GTData> loadGroundTruth(const std::string& pose_file, const std::string& time_file);
        gtsam::Pose3 getClosestGTPose(const std::vector<GTData>& gt_list, double target_time);
        int getClosestRadarIdx(const std::vector<double>& timestamps, double target_time);
        ImuBias getImuBias(const std::vector<ImuData>& all_imu_data, const double& static_duration);
    }
}