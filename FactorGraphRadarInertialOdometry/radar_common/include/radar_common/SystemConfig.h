#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "radar_common/Types.h"
#include "radar_registration/RadarICP.h"

namespace radar
{
    namespace common
    {
        using json = nlohmann::json;

        struct GraphConfig
        {
            bool use_2d_mode = false;
            bool gravity = 9.81;

            // IMU noise parameters
            double accel_noise_sigma = 2.45e-4;
            double gyro_noise_sigma = 8.73e-5;
            double accel_bias_rw_sigma_x = 1.0e-4;
            double accel_bias_rw_sigma_y = 1.0e-4;
            double accel_bias_rw_sigma_z = 1.0e-4;
            double gyro_bias_rw_sigma_x = 4.0e-5;
            double gyro_bias_rw_sigma_y = 4.0e-5;
            double gyro_bias_rw_sigma_z = 4.0e-5;

            // Prior Noise (Init Confidence)
            double prior_position_sigma = 0.01;
            double prior_orientation_sigma = 0.01; // in radians
            double prior_vel_sigma = 0.1;
            double prior_bias_accel_sigma = 1e-3;
            double prior_bias_gyro_sigma = 1e-3;

            // gtsam
            double isam2_relinearize_threshold = 0.1;
            size_t isam2_relinearize_skip = 10;
            double keyframe_rate = 0.01; // Hz - rate of keyframes
            size_t gt_skip_rate = 10;
            size_t update_rate = 10;
            double preinegration_sig_x = 0.011525434466888276;
            double preinegration_sig_y = 0.011525434466888276;
            double preinegration_sig_z = 0.011525434466888276;

            // Extrinsci Transforms (Base ->IMU)
            Transform T_base_imu;

            // ZUPT constraint (Zero velocity)
            double zero_vel_thresh = 0.05;

            // measurement noise of "GPS" (GT) pose
            double gt_sigma_position = 0.001;
            double gt_sigma_orientation = 0.002;
        };

        struct DatasetConfig
        {
            std::string base_dir;
            std::string calib_dir;
            std::string output_filename;
            std::string sensor;
            double static_duration = 0.0; // Duration to consider the system as static for initial calibration
        };

        struct SystemConfig
        {
            DatasetConfig dataset;
            EstimatorConfig reve; // REVE estimator config from radar_common/Types.h
            registration::ICPConfig icp; // ICP config from radar_registration/RadarICP.h
            GraphConfig graph;

            // Calibration paths relative to base_dir
            std::string tf_radar_path;
            std::string tf_imu_path;

            // Load configuration from a JSON file
            static SystemConfig load(const std::string& config_path)
            {
                std::ifstream file(config_path);
                if (!file.is_open())
                {
                    throw std::runtime_error("Could not open config file: " + config_path);
                }

                json config_json;
                file >> config_json;

                SystemConfig cfg;
                
                // Dataset config
                cfg.dataset.base_dir = config_json["dataset"]["base_dir"];
                cfg.dataset.calib_dir = config_json["dataset"]["calib_dir"];
                cfg.dataset.output_filename = config_json["dataset"]["output_filename"];
                cfg.dataset.static_duration = config_json.value("dataset", json::object()).value("static_duration", 0.0);
                cfg.dataset.sensor = config_json["dataset"]["sensor"];

                // Calibration
                cfg.tf_radar_path = config_json["calibration"]["tf_radar_path"];
                cfg.tf_imu_path = config_json["calibration"]["tf_imu_path"];

                // REVE config
                cfg.reve.ransac_iter = config_json["reve"]["ransac_iter"];
                cfg.reve.min_dist = config_json["reve"]["min_dist"];
                cfg.reve.max_dist = config_json["reve"]["max_dist"];
                cfg.reve.min_db = config_json["reve"]["min_db"];
                cfg.reve.min_elv = config_json["reve"]["min_elv"];
                cfg.reve.max_elv = config_json["reve"]["max_elv"];
                cfg.reve.min_azim = config_json["reve"]["min_azim"];
                cfg.reve.max_azim = config_json["reve"]["max_azim"];
                cfg.reve.inlier_thresh = config_json["reve"]["inlier_thresh"];
                cfg.reve.max_inlier_ratio = config_json["reve"]["max_inlier_ratio"];
                cfg.reve.zero_vel_thresh = config_json["reve"]["zero_vel_thresh"];
                cfg.reve.covariance_noise_floor = config_json["reve"]["covariance_noise_floor"];

                // ICP config
                cfg.icp.max_iterations = config_json["icp"]["max_iterations"];
                cfg.icp.max_correspondences_dist = config_json["icp"]["max_correspondences_dist"];
                cfg.icp.intensity_threshold = config_json["icp"]["intensity_threshold"];
                cfg.icp.transformation_epsilon = config_json["icp"]["transformation_epsilon"];
                cfg.icp.euclidean_fitness_epsilon = config_json["icp"]["euclidean_fitness_epsilon"];
                cfg.icp.dynamic_point_vel_thresh = config_json["icp"]["dynamic_point_vel_thresh"];
                cfg.icp.start_tx = config_json["icp"]["start_tx"];
                cfg.icp.start_ty = config_json["icp"]["start_ty"];
                cfg.icp.start_tz = config_json["icp"]["start_tz"];  
                cfg.icp.start_qx = config_json["icp"]["start_qx"];
                cfg.icp.start_qy = config_json["icp"]["start_qy"];
                cfg.icp.start_qz = config_json["icp"]["start_qz"];
                cfg.icp.start_qw = config_json["icp"]["start_qw"];

                // Graph config
                cfg.graph.use_2d_mode = config_json["graph"]["use_2d_mode"];
                cfg.graph.gravity = config_json["graph"].value("gravity", 9.81);
                cfg.graph.accel_noise_sigma = config_json["graph"]["noise"]["accel_sigma"];
                cfg.graph.gyro_noise_sigma = config_json["graph"]["noise"]["gyro_sigma"];
                cfg.graph.accel_bias_rw_sigma_x = config_json["graph"]["noise"]["accel_bias_rw_sigma_x"];
                cfg.graph.accel_bias_rw_sigma_y = config_json["graph"]["noise"]["accel_bias_rw_sigma_y"];
                cfg.graph.accel_bias_rw_sigma_z = config_json["graph"]["noise"]["accel_bias_rw_sigma_z"];
                cfg.graph.gyro_bias_rw_sigma_x = config_json["graph"]["noise"]["gyro_bias_rw_sigma_x"];
                cfg.graph.gyro_bias_rw_sigma_y = config_json["graph"]["noise"]["gyro_bias_rw_sigma_y"];
                cfg.graph.gyro_bias_rw_sigma_z = config_json["graph"]["noise"]["gyro_bias_rw_sigma_z"];
                cfg.graph.prior_position_sigma = config_json["graph"]["sigma"]["prior_position"];
                cfg.graph.prior_orientation_sigma = config_json["graph"]["sigma"]["prior_orientation"];
                cfg.graph.prior_vel_sigma = config_json["graph"]["sigma"]["prior_vel"];
                cfg.graph.prior_bias_accel_sigma = config_json["graph"]["sigma"]["prior_bias_accel"];
                cfg.graph.prior_bias_gyro_sigma = config_json["graph"]["sigma"]["prior_bias_gyro"];
                cfg.graph.isam2_relinearize_threshold = config_json["graph"]["isam2"]["relinearize_threshold"];
                cfg.graph.isam2_relinearize_skip = config_json["graph"]["isam2"]["relinearize_skip"];
                cfg.graph.keyframe_rate = config_json["graph"]["keyframe_rate"];
                cfg.graph.zero_vel_thresh = config_json["reve"]["zero_vel_thresh"];
                cfg.graph.gt_skip_rate = config_json["graph"]["gt_skip_rate"];
                cfg.graph.update_rate = config_json["graph"]["update_rate"];
                cfg.graph.gt_sigma_position = config_json["graph"]["noise"]["gt_sigma_position"];
                cfg.graph.gt_sigma_orientation = config_json["graph"]["noise"]["gt_sigma_orientation"];
                cfg.graph.preinegration_sig_x = config_json["graph"]["noise"]["preinegration_sig_x"];
                cfg.graph.preinegration_sig_y = config_json["graph"]["noise"]["preinegration_sig_y"];
                cfg.graph.preinegration_sig_z = config_json["graph"]["noise"]["preinegration_sig_z"];
                return cfg;
            }
        };
    }
}