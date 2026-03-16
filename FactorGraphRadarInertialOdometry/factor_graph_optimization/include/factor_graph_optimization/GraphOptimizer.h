#pragma once

#include "radar_common/Types.h"
#include "radar_common/SystemConfig.h"
#include "factor_graph_optimization/RadarVelocityFactor.h"
#include "factor_graph_optimization/DeltaYawFactor.h"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/ImuFactor.h> 
#include <gtsam/inference/Symbol.h>

namespace radar
{
    namespace optimization
    {
        class GraphOptimizer
        {
            private:
                // Config loader
                radar::common::GraphConfig config_;

                // ISAM2 solver
                gtsam::ISAM2 isam2_;

                // Containers for new factors and initial estimates to be added to graph
                gtsam::NonlinearFactorGraph new_factors_;
                gtsam::Values new_values_;

                bool use_2d_mode_ = true; // If true, only optimize x,y,theta and set z=0 for all poses
                std::vector<double> timestamp_history_;
                
                // Object to handle IMU preintegration
                std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrated_imu_;
                double bias_acc_sigma_;
                double bias_omega_sigma_;

                // Counter to keep track of node indices
                uint64_t current_node_idx_ = 0;
                uint64_t last_radar_pose_key_ = 0;
                int update_counter_ = 0;


                // The most recently optimized states
                gtsam::Pose3 current_pose_;
                gtsam::Vector3 current_velocity_;
                gtsam::imuBias::ConstantBias current_bias_;
                gtsam::SharedNoiseModel planar_pose_noise_;
                gtsam::SharedNoiseModel planar_vel_noise_;
                double current_timestamp_;

                // Helper function to attempt optimization and handle indeterminant system exceptions by adding pseudo-priors
                void attemptOptimization();

            public:
                GraphOptimizer(const radar::common::GraphConfig& config); // constructor
                ~GraphOptimizer() = default; // destructor

                /*
                @brief Set whether to use 2D mode (x,y,theta) or full 3D mode for pose optimization
                */
                void set2DMode(bool enabled) { use_2d_mode_= enabled; }

                /*
                @brief Initialize graph with priors at t=0
                */
                void initialize(const gtsam::Pose3& initial_pose,
                                const gtsam::Vector3& initial_velocity,
                                const gtsam::imuBias::ConstantBias& initial_bias,
                                const double& initial_timestamp,
                                const double& estimated_gravity);
                
                /*
                @brief Add new synchronized frame and optimize
                */
                void addFrame(const radar::common::OptimizationFrame& frame);

                // Getters
                gtsam::Pose3 getCurrentPose() const { return current_pose_; }
                gtsam::Vector3 getCurrentVelocity() const { return current_velocity_; }

                /*
                @brief Save the full optimized trajectory to a text file for analysis/visualization
                 Format: timestamp, x, y, z, roll, pitch, yaw
                */
                void saveFullTrajectory(const std::string& filename);

                // Exposing ISAM2 graph and values for visualization/debugging purposes
                const gtsam::ISAM2& getISAM2() const { return isam2_; }
        };
    }
}