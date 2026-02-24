#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>

namespace radar
{
    namespace optimization
    {
        /*
        @brief Custom GTSAM factor to fuse Body frame velocity from REVE
        Connects to Pose3 node to transform the measurement and estimate 
        into same frame before calculating the error.
        */
       class RadarVelocityFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>
       {
        private:
            // measurement from REVE algorithm
            gtsam::Vector3 measured_vel_body_;

        public:
            // constructor
            RadarVelocityFactor(gtsam::Key pose_key, gtsam::Key vel_key,
                                const gtsam::Vector3& measured_vel_body,
                                const gtsam::SharedNoiseModel& model) : 
                                gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector3>(model, pose_key, vel_key),
                                measured_vel_body_(measured_vel_body) {}

            // Destructor
            virtual ~RadarVelocityFactor() {}

            /*
            @brief The core error calculation
            Error = (Predicted velocity in Body Frame) - (Measured Velocity in Body Frame)
            */
            gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                                        const gtsam::Vector3& vel_global,
                                        boost::optional<gtsam::Matrix&> H1 = boost::none,
                                        boost::optional<gtsam::Matrix&> H2 = boost::none) const override
            {
                // Temp matrices for chain rule
                gtsam::Matrix36 H_rot_wrt_pose;
                gtsam::Matrix33 H_unrot_wrt_rot;
                gtsam::Matrix33 H_unrot_wrt_vel;

                // Extracting rotation and get jacobian of Rotation wrt Pose
                gtsam::Rot3 R_global = pose.rotation(H1 ? &H_rot_wrt_pose : 0);

                // Unrotate velocity and get Jacobians of unrotate wrt Rotation and Velocity
                gtsam::Vector3 predicted_vel_body = R_global.unrotate(vel_global, 
                                                                      H1 ? &H_unrot_wrt_rot : 0,
                                                                      H2 ? &H_unrot_wrt_vel : 0);

                if (H1) *H1 = H_unrot_wrt_rot * H_rot_wrt_pose; // Matrix multiplication (3x3 * 3x6 = 3x6)
                if (H2) *H2 = H_unrot_wrt_vel; // (3x3)

                return predicted_vel_body - measured_vel_body_;
            }
            
       };
    }
}