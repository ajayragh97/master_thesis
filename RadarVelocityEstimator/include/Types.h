#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace radar_estimator
{
    /*
    @brief A single radar target detection
    */
    struct RadarTarget
    {
        Eigen::Vector3d position_cart; // x, y, z
        Eigen::Vector3d position_sph; // range, azimuth, elevation
        Eigen::Vector3d direction_vector;
        double doppler_velocity;
        double snr_db;
    };

    /*
    @brief Pointclouds formed from RadarTargets
    */
    using PointCloud = std::vector<RadarTarget>;

    /*
    @brief Config parameters for the ego velocity estimator
    */
    struct EstimatorConfig
    {
        int ransac_iter = 100;
        double min_dist = 0.5;
        double max_dist = 10.0;
        double min_db = 50;
        double min_elv = -45;
        double max_elv = 45;
        double min_azim = -45;
        double max_azim = 45;
        double inlier_thresh = 0.2;
        double max_inlier_ratio = 0.9;
        double zero_vel_thresh = 0.05;
    };

    /*
    @brief Velocity estimated by the REVE algorith
    */
    struct VelocityEstimate
    {
        bool success = false;
        Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        int inlier_ratio;
    };


    /*
    @brief Static coordinate transform data
    */
    struct Transform
    {
        Eigen::Vector3d translation;
        Eigen::Quaterniond rotation;
    };

    /*
    @brief A single IMU measurement
    */
   struct ImuData
   {
    double timestamp;
    Eigen::Vector3d linear_acceleration;
    Eigen::Vector3d angular_velocity;
   };
   

    
}