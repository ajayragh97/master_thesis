#pragma once

#include "radar_common/Types.h"
#include "factor_graph_optimization/RadarVelocityFactor.h"
#include "factor_graph_optimization/DeltaYawFactor.h"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/navigation/PreintegratedImuMeasurements.h>
#include <gtsam/inference/Symbol.h>

namespace radar
{
    namespace optimization
    {
        class GraphOptimizer
        {
            private:
                // ISAM2 solver
                gtsam::ISAM2 isam2_;

                // Containers for new factors and initial estimates to be added to graph
                gtsam::NonlinearFactorGraph new_factors_;
                gtsam::Values new_values_;

                // Object to handle IMU preintegration
                std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrated_imu_;

                // Counter to keep track of current node index
                uint64_t current_node_idx_ = 0;

                // The most recently optimized states
                gtsam::Pose3 current_pose_;
                gtsam::Vector3 current_velocity_;
                gtsam::imuBias::ConstantBias current_bias_;
                double current_timestamp_;

            public:
                GraphOptimizer(); // constructor
                ~GraphOptimizer() = default; // destructor

                /*
                @brief Initialize graph with priors at t=0
                */
                void initialize(const gtsam::Pose3& initial_pose,
                                const gtsam::Vector3& initial_velocity,
                                const gtsam::imuBias::ConstantBias& initial_bias,
                                const double& initial_timestamp);
                
                /*
                @brief Add new synchronized frame and optimize
                */
                void addFrame(const radar::common::OptimizationFrame& frame);

                // Getters
                gtsam::Pose3 getCurrentPose() const { return current_pose_; }
                gtsam::Vector3 getCurrentVelocity() const { return current_velocity_; }
        };
    }
}