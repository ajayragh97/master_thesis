#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/base/numericalDerivative.h>

namespace radar
{
    namespace optimization
    {
        /*
        @brief Custom GTSAM factor that connects between twon Pose nodes.
        This factor checks the error between predicted delta Yaw angle between the nodes and 
        the measured Yaw angle between the nodes.
        */
        class DeltaYawFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
        {
            private:
                double measured_delta_yaw_;
            
            public:
                // Constructor
                DeltaYawFactor(gtsam::Key pose_prev_key, gtsam::Key pose_curr_key,
                               double measured_delta_yaw,
                               const gtsam::SharedNoiseModel& model) :
                               gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, pose_prev_key, pose_curr_key),
                               measured_delta_yaw_(measured_delta_yaw) {}

                // Destructor
                virtual ~DeltaYawFactor() {}

                gtsam::Vector evaluateError(const gtsam::Pose3& pose_prev,
                                            const gtsam::Pose3& pose_curr,
                                            boost::optional<gtsam::Matrix&> H1 = boost::none,
                                            boost::optional<gtsam::Matrix&> H2 = boost::none) const override
                {

                    // Lambda function to compute error
                    auto compute_err = [this](const gtsam::Pose3& p_prev, const gtsam::Pose3& p_curr) -> gtsam::Vector 
                    {
                        gtsam::Rot3 rot_delta = p_prev.rotation().between(p_curr.rotation());
                        double error_value = gtsam::Rot2::fromAngle(rot_delta.yaw() - measured_delta_yaw_).theta();
                        return (gtsam::Vector(1) << error_value).finished();
                    };

                    // Using gtsam numerical derivative tool for Jacobians
                    if (H1)
                    {
                        *H1 = gtsam::numericalDerivative21<gtsam::Vector, gtsam::Pose3, gtsam::Pose3>(compute_err, pose_prev, pose_curr, 1e-5);
                    }

                    if (H2)
                    {
                        *H2 = gtsam::numericalDerivative22<gtsam::Vector, gtsam::Pose3, gtsam::Pose3>(compute_err, pose_prev, pose_curr, 1e-5);
                    }

                    return compute_err(pose_prev, pose_curr);
                   
                }
        };
    }
}