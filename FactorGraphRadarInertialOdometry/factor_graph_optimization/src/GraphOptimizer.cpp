#include "factor_graph_optimization/GraphOptimizer.h"
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/ImuFactor.h>

using gtsam::symbol_shorthand::X; // Pose
using gtsam::symbol_shorthand::V; // Velocity
using gtsam::symbol_shorthand::B; // Bias

namespace radar
{
    namespace optimization
    {
        GraphOptimizer::GraphOptimizer() 
        {
            gtsam::ISAM2Params params;
            params.relinearizeThreshold = 0.1;
            params.relinearizeSkip = 10;
            isam2_ = gtsam::ISAM2(params);

            // Setting up IMU pre integration parameters
            auto imu_params = gtsam::PreintegrationParams::MakeSharedU(9.81);

            // loading IMU covariance. For now hard coding, need to change
            imu_params->accelerometerCovariance = gtsam::Matrix33::Identity() * 1e-3;
            imu_params->gyroscopeCovariance = gtsam::Matrix33::Identity() * 1e-4;
            imu_params->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8;

            preintegrated_imu_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params,
                                                                                       gtsam::imuBias::ConstantBias());
        }

        void GraphOptimizer::initialize(const gtsam::Pose3& initial_pose,
                                        const gtsam::Vector3& initial_velocity,
                                        const gtsam::imuBias::ConstantBias& initial_bias,
                                        const double& initial_timestamp)
        {
            // Creating priors 
            auto pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05).finished());
            auto vel_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.1, 0.1, 0.1).finished());
            auto bias_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());

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
        }

        void GraphOptimizer::addFrame(const radar::common::OptimizationFrame& frame)
        {
            // update nose index
            uint64_t prev_idx = current_node_idx_;
            current_node_idx_++;
            uint64_t curr_idx = current_node_idx_;

            // preintegrate IMU measurements
            for (size_t i = 0; i < frame.imu_measurements.size(); ++i)
            {
                const radar::common::ImuData& curr_imu_frame = frame.imu_measurements[i];
                double dt = 0;

                if (i == 0)
                {
                    dt = curr_imu_frame.timestamp - curr_timestamp_;
                }
                else
                {
                    dt = curr_imu_frame.timestamp - frame.imu_measurements[i-1].timestamp;
                }
                preintegrated_imu_->integrateMeasurement(curr_imu_frame.linear_acceleration, 
                                                        curr_imu_frame.angular_velocity,
                                                        dt);                
            }

            // adding imu factor to new_factors_
            new_factors_.add(gtsam::ImuFactor(X(prev_idx), V(prev_idx), X(curr_idx), V(curr_idx), B(prev_idx), *preintegrated_imu_))

            // adding IMU bias random walk to new_factors_
            auto bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());
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
                auto delta_yaw_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << yaw_sigma));
                new_factors_.add(DeltaYawFactor(X(prev_idx), X(curr_idx), frame.delta_yaw, delta_yaw_noise_model));
            }

            // predict new values


            gtsam::NavState predicted_state = preintegrated_imu_->predict(NavState(current_pose_, current_velocity_), current_bias_);
            new_values_.insert(X(curr_idx), predicted_state.pose());
            new_values_.insert(V(curr_idx), predicted_state.velocity());
            new_values_.insert(B(curr_idx), current_bias_);

            // update ISAM2
            isam2_.update(new_factors_, new_values_);
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
    }
}