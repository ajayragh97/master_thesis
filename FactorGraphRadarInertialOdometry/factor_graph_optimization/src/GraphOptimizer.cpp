#include "factor_graph_optimization/GraphOptimizer.h"
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using gtsam::symbol_shorthand::X; // Pose
using gtsam::symbol_shorthand::V; // Velocity
using gtsam::symbol_shorthand::B; // Bias

namespace radar
{
    namespace optimization
    {
        GraphOptimizer::GraphOptimizer(const radar::common::GraphConfig& config) : config_(config)
        {

            gtsam::ISAM2Params params;
            params.relinearizeThreshold = config_.isam2_relinearize_threshold;
            params.relinearizeSkip = config_.isam2_relinearize_skip;
            isam2_ = gtsam::ISAM2(params);

            // Setup 2D Constraint Models
            /*Using a Diagonal noise model
            GTSAM Pose3 vector order: [Roll, Pitch, Yaw, X, Y, Z]
            Tight constraint 
            Loose constraint
            */  
            double tight = 1e-4;
            double loose = 100.0;
            // Constrain Roll, Pitch, Z. Let Yaw, X, Y float.
            planar_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << tight, tight, loose, loose, loose, tight).finished());
            // Constrain Velocity Z. Let Vx, Vy float.
            planar_vel_noise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << loose, loose, tight).finished());
        }

        void GraphOptimizer::initialize(const gtsam::Pose3& initial_pose,
                                        const gtsam::Vector3& initial_velocity,
                                        const gtsam::imuBias::ConstantBias& initial_bias,
                                        const double& initial_timestamp,
                                        const double& estimated_gravity,
                                        const radar::common::ImuBias& imu_bias)
        {
            // setup preintegrator with estimated gravity
            auto imu_params = gtsam::PreintegrationParams::MakeSharedU(estimated_gravity);
            gtsam::Point3 t(config_.T_base_imu.translation);
            gtsam::Rot3 R(gtsam::Rot3::Roll(M_PI));
            imu_params->setBodyPSensor(gtsam::Pose3(R, t));

            // loading IMU covariance
            if (imu_bias.std_acc != Eigen::Vector3d::Zero())
            {
                std::cout << "seting preintegrator covariances" << std::endl;
                // Covariance Matrix Accelerometer
                Eigen::DiagonalMatrix<double, 3> cov_acc(pow(imu_bias.std_acc.x(), 2),
                                                         pow(imu_bias.std_acc.y(), 2), 
                                                         pow(imu_bias.std_acc.z(), 2));
                
                // Covariance Matrix Gyroscope
                Eigen::DiagonalMatrix<double, 3> cov_gyro(pow(imu_bias.std_gyro.x(), 2),
                                                         pow(imu_bias.std_gyro.y(), 2), 
                                                         pow(imu_bias.std_gyro.z(), 2));

                // Covariance Matrix Integration
                Eigen::DiagonalMatrix<double, 3> cov_integration(pow(config_.preinegration_sig_x, 2),
                                                         pow(config_.preinegration_sig_y, 2), 
                                                         pow(config_.preinegration_sig_z, 2));
                
                imu_params->setAccelerometerCovariance(cov_acc);
                imu_params->setGyroscopeCovariance(cov_gyro);
                imu_params->setIntegrationCovariance(cov_integration);
                imu_params->setUse2ndOrderCoriolis(false);    
            }
            else
            {
                imu_params->accelerometerCovariance = gtsam::Matrix33::Identity() * std::pow(config_.accel_noise_sigma, 2);
                imu_params->gyroscopeCovariance = gtsam::Matrix33::Identity() * std::pow(config_.gyro_noise_sigma, 2);
                imu_params->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8;
            }
            
            
            preintegrated_imu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params,
                                                                                       initial_bias);

            // Creating priors pose is rotation first, translation second in gtsam
            auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  config_.prior_orientation_sigma, 
                                                                                        config_.prior_orientation_sigma, 
                                                                                        config_.prior_orientation_sigma,
                                                                                        config_.prior_position_sigma,
                                                                                        config_.prior_position_sigma, 
                                                                                        config_.prior_position_sigma).finished());

            auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) <<   config_.prior_vel_sigma, 
                                                                                        config_.prior_vel_sigma, 
                                                                                        config_.prior_vel_sigma).finished());

            auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  config_.prior_bias_accel_sigma,
                                                                                        config_.prior_bias_accel_sigma,
                                                                                        config_.prior_bias_accel_sigma,
                                                                                        config_.prior_bias_gyro_sigma,
                                                                                        config_.prior_bias_gyro_sigma,
                                                                                        config_.prior_bias_gyro_sigma).finished());

            // Adding prior factors to graph at index 0
            new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), initial_pose, pose_noise));
            new_factors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), initial_velocity, vel_noise));
            new_factors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), initial_bias, bias_noise));

            // Adding initial values
            new_values_.insert(X(0), initial_pose);
            new_values_.insert(V(0), initial_velocity);
            new_values_.insert(B(0), initial_bias);

            // Update ISAM2 and state variables
            isam2_.update(new_factors_, new_values_);
            new_factors_.resize(0);
            new_values_.clear();

            current_pose_ = initial_pose;
            current_velocity_  = initial_velocity;
            current_bias_ = initial_bias;
            current_node_idx_ = 0;
            current_timestamp_ = initial_timestamp;
            last_radar_pose_key_ = 0;
            update_counter_ = 0;

            // Clear timestamp history and add initial timestamp
            timestamp_history_.clear();
            timestamp_history_.push_back(initial_timestamp);

            // update bias standard deviations (sigma) after init
            // config_.accel_bias_rw_sigma_x = imu_bias.std_acc.x();
            // config_.accel_bias_rw_sigma_y = imu_bias.std_acc.y();
            // config_.accel_bias_rw_sigma_z = imu_bias.std_acc.z();

            // config_.gyro_bias_rw_sigma_x = imu_bias.std_gyro.x();
            // config_.gyro_bias_rw_sigma_y = imu_bias.std_gyro.y();
            // config_.gyro_bias_rw_sigma_z = imu_bias.std_gyro.z();
        }

        void GraphOptimizer::addFrame(const radar::common::OptimizationFrame& frame)
        {

            // ==============================================================
            // SAFETY CHECK: Do not increment index if data is invalid
            // ==============================================================
            if (frame.imu_measurements.empty() && current_node_idx_ > 0) 
            {
                std::cerr << "WARNING: Empty IMU frame! Integrating blind gap to preserve graph chain." << std::endl;
                // We integrate 'zero' movement over the time gap to prevent GTSAM from crashing
                preintegrated_imu_->integrateMeasurement(
                    gtsam::Vector3::Zero(), gtsam::Vector3::Zero(), frame.timestamp - current_timestamp_);
            }


            // Ensure strictly positive time diff for random walk to prevent zero-sigma Information Matrix crashes
            double dt_frame = std::max(frame.timestamp - current_timestamp_, 1e-5);

            // update pose index
            uint64_t prev_idx = current_node_idx_;
            current_node_idx_++;
            uint64_t curr_idx = current_node_idx_;

            // Add timestamp to history
            timestamp_history_.push_back(frame.timestamp);

            // IMU Preintegration
            double last_t = current_timestamp_;
            for (const auto& imu : frame.imu_measurements)
            {
                // SAFEGUARD: Prevent out-of-order IMU messages from causing negative dt (which turns covariances into NaNs)
                if (imu.timestamp <= last_t) continue; 

                double dt = imu.timestamp - last_t;
                preintegrated_imu_ -> integrateMeasurement(imu.linear_acceleration, imu.angular_velocity, dt);
                last_t = imu.timestamp;
            }

            // IMPORTANT: Integrate the final gap between the last IMU message and the Radar timestamp
            double final_dt = frame.timestamp - last_t;
            if (final_dt > 1e-6 && !frame.imu_measurements.empty()) 
            {
                preintegrated_imu_->integrateMeasurement(
                    frame.imu_measurements.back().linear_acceleration, 
                    frame.imu_measurements.back().angular_velocity, 
                    final_dt);
            }

            

            // adding imu factor to new_factors_
            new_factors_.add(gtsam::ImuFactor(X(prev_idx), V(prev_idx), X(curr_idx), V(curr_idx), B(prev_idx), *preintegrated_imu_));

            // adding IMU bias random walk to new_factors_
            // scaled based on time elapsed to account for random walk variations            
            auto bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<config_.accel_bias_rw_sigma_x * std::sqrt(dt_frame),
                                                                                            config_.accel_bias_rw_sigma_y * std::sqrt(dt_frame),
                                                                                            config_.accel_bias_rw_sigma_z * std::sqrt(dt_frame),
                                                                                            config_.gyro_bias_rw_sigma_x * std::sqrt(dt_frame),
                                                                                            config_.gyro_bias_rw_sigma_y * std::sqrt(dt_frame),
                                                                                            config_.gyro_bias_rw_sigma_z * std::sqrt(dt_frame)).finished());
                                                                                            
            new_factors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(prev_idx), B(curr_idx), 
                                                                                gtsam::imuBias::ConstantBias(), 
                                                                                bias_noise_model));

            // =====================================
            // Sensor Measurements (GT, REVE, ICP)
            // =====================================

            // Ground truth "GPS" prior
            if (frame.has_gt_pose)
            {
                // Note gtsam by default uses rotation first then translation
                auto gt_meas_sigma = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<   config_.gt_sigma_orientation,
                                                                                                config_.gt_sigma_orientation,
                                                                                                config_.gt_sigma_orientation,
                                                                                                config_.gt_sigma_position,
                                                                                                config_.gt_sigma_position,
                                                                                                config_.gt_sigma_position).finished());
                new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(curr_idx), frame.gt_pose, gt_meas_sigma));
            }

            // adding velocity factor to new_factors_
            if (frame.has_reve_velocity)
            {
                auto velocity_covariance = gtsam::noiseModel::Gaussian::Covariance(frame.reve_velocity_body.covariance);                                 
                new_factors_.add(RadarVelocityFactor(X(curr_idx), V(curr_idx), frame.reve_velocity_body.linear_velocity, velocity_covariance));
            }
            
            // Radar ICP transform factor
            if (frame.has_icp && last_radar_pose_key_ != 0)
            {
                gtsam::Rot3 icp_rot(frame.T_body.block<3,3>(0,0));
                gtsam::Point3 icp_trans(frame.T_body(0,3), frame.T_body(1,3), frame.T_body(2, 3));
                gtsam::Pose3 relative_icp_pose(icp_rot, icp_trans);

                // standard deviation  = sqrt(MSE), adding a floor to prevent overconfidence
                double icp_std = std::max(sqrt(frame.icp_fitness), 0.01);
                auto icp_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<0.1, // roll (rad)(radar icp cant be fully trusted)
                                                                                        0.1, // pitch (rad)
                                                                                        icp_std *2, // yaw (rad)
                                                                                        icp_std, // x(m)
                                                                                        icp_std, // y(m)
                                                                                        0.1).finished()); // z(m) variance in z is high
                                
                new_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(X(last_radar_pose_key_), X(curr_idx), relative_icp_pose, icp_noise));
            }

            // if in 2d mode
            if (use_2d_mode_)
            {
                // adding delta yaw factor
                if (frame.has_delta_yaw)
                {
                    double yaw_sigma = std::max(frame.delta_yaw_fitness, 1e-4); 
                    auto delta_yaw_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << yaw_sigma).finished());
                    new_factors_.add(DeltaYawFactor(X(last_radar_pose_key_), X(curr_idx), frame.delta_yaw, delta_yaw_noise_model));
                    last_radar_pose_key_ = curr_idx;
                }

                // 1. Constrain Pose: Force Roll=0, Pitch=0, Z=Constant
                // We create a "Target" pose that has 0 roll/pitch and the initial Z height.
                // The loose noise model ignores the X,Y,Yaw diffs, but penalizes Z,Roll,Pitch diffs.

                gtsam::Pose3 planar_target(gtsam::Rot3::Ypr(0.0, 0.0, 0.0), 
                                   gtsam::Point3(0.0, 0.0, current_pose_.z())); // Keep Z at initial height

                new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(curr_idx), planar_target, planar_pose_noise_));

                // 2. Constrain Velocity: Force Vz=0
                gtsam::Vector3 vel_target(0.0, 0.0, 0.0);
                new_factors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(curr_idx), vel_target, planar_vel_noise_));
            }

            if (frame.has_icp || curr_idx == 0) last_radar_pose_key_ = curr_idx;

            // State prediction and preintegration reset
            // Predict new state using the tracking state from the LAST frame
            gtsam::NavState predicted_state = preintegrated_imu_->predict(gtsam::NavState(current_pose_, current_velocity_), current_bias_);
            new_values_.insert(X(curr_idx), predicted_state.pose());
            new_values_.insert(V(curr_idx), predicted_state.velocity());
            new_values_.insert(B(curr_idx), current_bias_);

            // Update tracking state to keep prediction chain connected
            current_pose_ = predicted_state.pose();
            current_velocity_ = predicted_state.velocity();

            // reset the preintegrator so the NEXT frame strictly evaluates[curr_idx -> curr_idx+1]
            preintegrated_imu_->resetIntegrationAndSetBias(current_bias_);

            // ===================================
            // ISAM2 Batch optimization
            // ===================================
            update_counter_++;
            if (update_counter_ >= config_.isam2_relinearize_skip)
            {
                attemptOptimization();

                // Fix drift by fetching the globally optimized states
                gtsam::Values result = isam2_.calculateEstimate();
                current_pose_ = result.at<gtsam::Pose3>(X(curr_idx));
                current_velocity_ = result.at<gtsam::Vector3>(V(curr_idx));
                current_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(curr_idx));

                // The IMU bias was likely updated by the optimization. Re-apply it to the preintegrator.
                preintegrated_imu_->resetIntegrationAndSetBias(current_bias_);
                update_counter_ = 0;
            }

            current_timestamp_ = frame.timestamp;
        }

        void GraphOptimizer::saveFullTrajectory(const std::string& filename)
        {
            std::cout << "Optimizing full graph and saving to " << filename << "..." << std::endl;

            std::ofstream outFile(filename);
            if (!outFile.is_open()) {
                std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
                return;
            }
            
            // TUM Format precision
            outFile << std::fixed << std::setprecision(6);

            // 1. Get the BEST estimate for ALL variables in the graph
            gtsam::Values full_estimate = isam2_.calculateEstimate();

            // 2. Iterate through our history
            for (size_t i = 0; i < timestamp_history_.size(); ++i)
            {
                gtsam::Key key = X(i); // Pose key for frame i

                // Check if the key exists (it should, but safety first)
                if (full_estimate.exists(key))
                {
                    gtsam::Pose3 pose = full_estimate.at<gtsam::Pose3>(key);
                    gtsam::Matrix covariance = isam2_.marginalCovariance(key);

                    double c_xx = covariance(3, 3);
                    double c_yy = covariance(4, 4);
                    double c_zz = covariance(5, 5);
                    double c_xy = covariance(3, 4); 
                    double c_xz = covariance(3, 5); 
                    double c_yz = covariance(4, 5);
                    
                    // Extended TUM Format: timestamp x y z qx qy qz qw c_xx c_yy c_zz c_xy c_xz c_yz
                    // Note: GTSAM quaternions are (w, x, y, z), Eigen is (x, y, z, w)
                    // pose.rotation().toQuaternion() returns an Eigen quaternion.
                    auto q = pose.rotation().toQuaternion();

                    outFile << timestamp_history_[i] << " "
                            << pose.translation().x() << " "
                            << pose.translation().y() << " "
                            << pose.translation().z() << " "
                            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                            << c_xx << " " << c_yy << " " << c_zz << " " 
                            << c_xy << " " << c_xz << " " << c_yz << "\n";
                }
            }
            
            outFile.close();
            std::cout << "Full trajectory saved successfully." << std::endl;
        }

        void GraphOptimizer::attemptOptimization()
        {
            // Try to update normally
            isam2_.update(new_factors_, new_values_);
            // cleanup after successful optimization
            new_factors_.resize(0);
            new_values_.clear();                  
        }

        
    }
}