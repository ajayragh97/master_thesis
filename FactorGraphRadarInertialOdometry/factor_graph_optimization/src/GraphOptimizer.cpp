#include "factor_graph_optimization/GraphOptimizer.h"
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>

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

            // Setting up IMU pre integration parameters
            auto imu_params = gtsam::PreintegrationParams::MakeSharedU(config_.gravity);
            gtsam::Point3 t(config_.T_base_imu.translation);
            gtsam::Rot3 R(config_.T_base_imu.rotation);
            imu_params->setBodyPSensor(gtsam::Pose3(R, t));

            // loading IMU covariance
            imu_params->accelerometerCovariance = gtsam::Matrix33::Identity() * std::pow(config_.accel_noise_sigma, 2);
            imu_params->gyroscopeCovariance = gtsam::Matrix33::Identity() * std::pow(config_.gyro_noise_sigma, 2);
            imu_params->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8;
            
            preintegrated_imu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params,
                                                                                       gtsam::imuBias::ConstantBias());

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
                                        const double& initial_timestamp)
        {
            // Creating priors 
            auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  config_.prior_position_sigma,
                                                                                        config_.prior_position_sigma, 
                                                                                        config_.prior_position_sigma, 
                                                                                        config_.prior_orientation_sigma, 
                                                                                        config_.prior_orientation_sigma, 
                                                                                        config_.prior_orientation_sigma).finished());

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

            // Clear timestamp history and add initial timestamp
            timestamp_history_.clear();
            timestamp_history_.push_back(initial_timestamp);
        }

        void GraphOptimizer::addFrame(const radar::common::OptimizationFrame& frame)
        {

            // ==============================================================
            // SAFETY CHECK FIRST: Do not increment index if data is invalid
            // ==============================================================
            if (frame.imu_measurements.empty()) 
            {
                std::cerr << "CRITICAL: Frame at " << std::fixed << std::setprecision(6) 
                        << frame.timestamp << " has NO IMU measurements. Skipping graph update." << std::endl;
                return; 
            }

            // update pose index
            uint64_t prev_idx = current_node_idx_;
            current_node_idx_++;
            uint64_t curr_idx = current_node_idx_;

            // Add timestamp to history
            timestamp_history_.push_back(frame.timestamp);

            // preintegrate IMU measurements
            if (frame.imu_measurements.empty()) 
            {
                std::cerr << "CRITICAL: No IMU measurements passed to optimizer! Cannot integrate." << std::endl;
                return; // Safely abort this frame
            }

            for (size_t i = 0; i < frame.imu_measurements.size(); ++i)
            {
                const radar::common::ImuData& curr_imu_frame = frame.imu_measurements[i];
                double dt = 0;

                if (i == 0)
                {
                    dt = curr_imu_frame.timestamp - current_timestamp_;
                }
                else
                {
                    dt = curr_imu_frame.timestamp - frame.imu_measurements[i-1].timestamp;
                }
                if (dt <= 1e-4) 
                {
                    // dt = 1e-4; // Force a tiny positive delta
                    std::cerr << "WARNING: Too small dt between IMU measurements. SKIPPING FRAME.." << std::endl;
                    continue;
                }

                preintegrated_imu_->integrateMeasurement(curr_imu_frame.linear_acceleration, 
                                                        curr_imu_frame.angular_velocity,
                                                        dt);                
            }

            // IMPORTANT: Integrate the final gap between the last IMU message and the Radar timestamp
            double final_dt = frame.timestamp - frame.imu_measurements.back().timestamp;
            if (final_dt > 0.0) 
            {
                preintegrated_imu_->integrateMeasurement(
                    frame.imu_measurements.back().linear_acceleration, 
                    frame.imu_measurements.back().angular_velocity, 
                    final_dt);
            }

            // adding imu factor to new_factors_
            new_factors_.add(gtsam::ImuFactor(X(prev_idx), V(prev_idx), X(curr_idx), V(curr_idx), B(prev_idx), *preintegrated_imu_));

            // adding IMU bias random walk to new_factors_
            
            auto bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<config_.accel_bias_rw_sigma,
                                                                                            config_.accel_bias_rw_sigma,
                                                                                            config_.accel_bias_rw_sigma,
                                                                                            config_.gyro_bias_rw_sigma,
                                                                                            config_.gyro_bias_rw_sigma,
                                                                                            config_.gyro_bias_rw_sigma).finished());
                                                                                            
            new_factors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(prev_idx), B(curr_idx), 
                                                                                gtsam::imuBias::ConstantBias(), 
                                                                                bias_noise_model));

            // adding velocity factor to new_factors_
            if (frame.has_reve_velocity)
            {
                auto velocity_covariance = gtsam::noiseModel::Gaussian::Covariance(frame.reve_velocity_body.covariance);                                 
                new_factors_.add(RadarVelocityFactor(X(curr_idx), V(curr_idx), frame.reve_velocity_body.linear_velocity, velocity_covariance));
            }
            
            // adding delta yaw factor
            if (frame.has_delta_yaw)
            {
                double yaw_sigma = std::max(frame.delta_yaw_fitness, 1e-4); 
                auto delta_yaw_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << yaw_sigma).finished());
                new_factors_.add(DeltaYawFactor(X(last_radar_pose_key_), X(curr_idx), frame.delta_yaw, delta_yaw_noise_model));
                last_radar_pose_key_ = curr_idx;
            }

            // Apply 2D constraints if in 2D mode
            if (use_2d_mode_)
            {
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
            // predict new values

            gtsam::NavState predicted_state = preintegrated_imu_->predict(gtsam::NavState(current_pose_, current_velocity_), current_bias_);
            new_values_.insert(X(curr_idx), predicted_state.pose());
            new_values_.insert(V(curr_idx), predicted_state.velocity());
            new_values_.insert(B(curr_idx), current_bias_);

            // update ISAM2
            attemptOptimization();
            new_factors_.resize(0);
            new_values_.clear();

            // update current states
            gtsam::Values result = isam2_.calculateEstimate();
            current_pose_ = result.at<gtsam::Pose3>(X(curr_idx));
            current_velocity_ = result.at<gtsam::Vector3>(V(curr_idx));
            current_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(curr_idx));
            current_timestamp_ = frame.timestamp;
            
            // Reset IMU integrator
            preintegrated_imu_->resetIntegrationAndSetBias(current_bias_);         

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
                    
                    // TUM Format: timestamp x y z qx qy qz qw
                    // Note: GTSAM quaternions are (w, x, y, z), Eigen is (x, y, z, w)
                    // pose.rotation().toQuaternion() returns an Eigen quaternion.
                    auto q = pose.rotation().toQuaternion();

                    outFile << timestamp_history_[i] << " "
                            << pose.translation().x() << " "
                            << pose.translation().y() << " "
                            << pose.translation().z() << " "
                            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
                }
            }
            
            outFile.close();
            std::cout << "Full trajectory saved successfully." << std::endl;
        }

        void GraphOptimizer::attemptOptimization()
        {
            // Try to update normally
            try
            {
                isam2_.update(new_factors_, new_values_);
                // cleanup after successful optimization
                new_factors_.resize(0);
                new_values_.clear();
            }
            catch(const gtsam::IndeterminantLinearSystemException& e)
            {
                std::cerr << "[GraphOptimizer] Indeterminant System detected at Key " 
                  << gtsam::DefaultKeyFormatter(e.nearbyVariable()) << ". Attempting Recovery..." << std::endl;
                
                new_values_.clear();
                
                // Recovery strategy: The graph is floating. we have to anchor the current variables
                // Using the current estimates as "pseudo-priors" to anchor the graph and break the indeterminacy

                uint64_t curr = current_node_idx_;
                
                // 1. Anchor Pose (Weakly pin XYZ, strongly pin RP if in 2D mode)
                auto pose_noise = use_2d_mode_ ? planar_pose_noise_ : gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  config_.prior_position_sigma, 
                                                                                                                                config_.prior_position_sigma, 
                                                                                                                                config_.prior_position_sigma, 
                                                                                                                                config_.prior_orientation_sigma, 
                                                                                                                                config_.prior_orientation_sigma, 
                                                                                                                                config_.prior_orientation_sigma).finished());

                new_factors_.add(gtsam::PriorFactor<gtsam::Pose3>(X(curr), current_pose_, pose_noise));

                // 2. Anchor Velocity (If in 2D mode, only anchor Vz)
                auto vel_noise = use_2d_mode_ ? planar_vel_noise_ : gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) <<   config_.prior_vel_sigma, 
                                                                                                            config_.prior_vel_sigma, 
                                                                                                            config_.prior_vel_sigma).finished());
                new_factors_.add(gtsam::PriorFactor<gtsam::Vector3>(V(curr), current_velocity_, vel_noise));

                // 3. Anchor Bias
                auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  config_.prior_bias_accel_sigma,
                                                                                            config_.prior_bias_accel_sigma,
                                                                                            config_.prior_bias_accel_sigma,
                                                                                            config_.prior_bias_gyro_sigma,
                                                                                            config_.prior_bias_gyro_sigma,
                                                                                            config_.prior_bias_gyro_sigma).finished());
                new_factors_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(curr), current_bias_, bias_noise));

                // Retry optimization
                try
                {
                    gtsam::Values empty_values; 
                    isam2_.update(new_factors_, empty_values);
                    std::cout << "[GraphOptimizer] Recovery successful. Graph optimized." << std::endl;
                }
                catch(const gtsam::IndeterminantLinearSystemException& e)
                {
                    std::cerr << "[GraphOptimizer] Recovery attempt failed at Key " 
                              << gtsam::DefaultKeyFormatter(e.nearbyVariable()) 
                              << ". Graph remains indeterminant. This frame will be skipped." << std::endl;
                    new_factors_.resize(0); // Clear the pseudo-priors to avoid polluting future optimizations
                    new_values_.clear();
                    return; // Skip this frame's update
                }
            }

            new_factors_.resize(0);
            new_values_.clear();           
            
        }

        
    }
}