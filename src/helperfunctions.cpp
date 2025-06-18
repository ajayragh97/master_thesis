#include "../include/master_thesis/helperfunctions.hpp"
#include <cmath>
#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iostream> 
#include <Eigen/Dense> 
#include <Eigen/Eigenvalues> 


namespace helperfunctions
{

    const float GOLDEN_RATIO_CONJUGATE = 0.61803398875f; 
    const float HUE_START_OFFSET = 0.25f;

    std_msgs::msg::ColorRGBA hsvToRgb(float h, float s, float v, float a) 
    {
        std_msgs::msg::ColorRGBA rgb_color;
        rgb_color.a = a;

        if (s < 1e-6f) { // Achromatic (grey), compare with a small epsilon
            rgb_color.r = v;
            rgb_color.g = v;
            rgb_color.b = v;
            return rgb_color;
        }

        // h is assumed to be in [0, 1)
        int i = static_cast<int>(h * 6.0f);
        float f = (h * 6.0f) - i; // Fractional part of h*6
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));

        i %= 6; // Ensure i is 0 to 5

        switch (i) {
            case 0: rgb_color.r = v; rgb_color.g = t; rgb_color.b = p; break;
            case 1: rgb_color.r = q; rgb_color.g = v; rgb_color.b = p; break;
            case 2: rgb_color.r = p; rgb_color.g = v; rgb_color.b = t; break;
            case 3: rgb_color.r = p; rgb_color.g = q; rgb_color.b = v; break;
            case 4: rgb_color.r = t; rgb_color.g = p; rgb_color.b = v; break;
            // case 5:
            default: rgb_color.r = v; rgb_color.g = p; rgb_color.b = q; break;
        }
        return rgb_color;
    }

    std_msgs::msg::ColorRGBA generateColorFromGaussianID(int gaussian_id, float alpha) 
    {
        std_msgs::msg::ColorRGBA color;
        color.a = alpha;

        if (gaussian_id == -2) 
        { 
            color.r = 0.5f; 
            color.g = 0.5f;
            color.b = 0.5f;
        } 
        else 
        {

            float id_float_for_hue = static_cast<float>(std::abs(gaussian_id));

            // Calculate hue using the golden angle progression
            // Add a starting offset if you don't want ID 0 (or low IDs) to always be reddish.
            float hue = std::fmod(HUE_START_OFFSET + id_float_for_hue * GOLDEN_RATIO_CONJUGATE, 1.0f);
            
            float saturation = 0.90f; 
            float value = 0.95f;      


            color = hsvToRgb(hue, saturation, value, alpha);
        }
        return color;
    }

    datastructures::ClusterMap create_dataframe(std::vector<datastructures::Point>& points)
    {
        datastructures::Point   temp_point;
        datastructures::ClusterMap cluster_data;

        for(size_t i = 0 ; i < points.size() ; ++i)
        {
            int id  =   points[i].cluster_id;
            
            if(id >= 0)
            {
                temp_point.x        =   points[i].x;
                temp_point.y        =   points[i].y;
                temp_point.z        =   points[i].z;
                temp_point.rcs      =   points[i].rcs;
                temp_point.vel      =   points[i].vel;
                temp_point.azimuth  =   points[i].azimuth;
                temp_point.range    =   points[i].range;

                cluster_data[id].points.push_back(temp_point);

            }
        }

        return cluster_data;
    }

    double calculate_angle_accuracy(const double& angle, const double& angle_min, const double& angle_max, const double& angle_var_min, const double& angle_var_max)
    {
        // linear relationship (y = mx + c). y = angle_accuracy, x = angle_measurement
        double m = (angle_var_max - angle_var_min) / (angle_max - angle_min);
        double c = angle_var_min - (m * angle_min);

        // now with m and c we can find the approximate measurement accuracy
        double accuracy = (m * std::abs(angle)) + c;

        return accuracy;     
    }

    void calculate_covariance_2d(datastructures::ClusterData& cluster_data)
    {
        double angle_max            =   1.0471975512;
        double angle_min            =   0.0;
        double angle_var_max        =   0.034906585;
        double angle_var_min        =   0.0087266463;
        double range_var            =   0.15;

        for(auto& point : cluster_data.points)
        {
            double azimuth_var      =   calculate_angle_accuracy(point.azimuth, angle_min, angle_max, angle_var_min, angle_var_max);

            // spherical variance
            double sigma_sq_r       =   pow(range_var, 2.0);
            double sigma_sq_theta   =   pow(azimuth_var, 2.0);

            Eigen::Matrix2d j;
            Eigen::Matrix2d sph_cov;

            j(0, 0) = cos(point.azimuth);
            j(0, 1) = (-point.range) * sin(point.azimuth);
            j(1, 0) = sin(point.azimuth);
            j(1, 1) = point.range * cos(point.azimuth);

            sph_cov(0, 0) = sigma_sq_r;
            sph_cov(0, 1) = 0;
            sph_cov(1, 0) = 0;
            sph_cov(1, 1) = sigma_sq_theta;

            Eigen::Matrix2d cart_cov    =   j * sph_cov * j.transpose();
            point.cov_2d                =   cart_cov;

        }
    }

    void calculate_covariance(datastructures::ClusterData& cluster_data)
    {
        double angle_max            =   1.0471975512;
        double angle_min            =   0.0;
        double angle_var_max        =   0.034906585;
        double angle_var_min        =   0.0087266463;
        double range_var            =   0.15;
        double elevation            =   0.0; // we only have 2d data

        for(auto& point : cluster_data.points)
        {
            double azimuth_var      =   calculate_angle_accuracy(point.azimuth, angle_min, angle_max, angle_var_min, angle_var_max);
            // double elevation_var=   calculate_angle_accuracy(point.elevation, angle_min, angle_max, angle_var_min, angle_var_max);
            double elevation_var    =   0.0;

            // spherical variance
            double sigma_sq_r       =   pow(range_var, 2.0);
            double sigma_sq_theta   =   pow(azimuth_var, 2.0);
            double sigma_sq_phi     =   pow(elevation_var, 2.0);

            // <---------------3D covariance------------------>
            // spherical covariance matrix
            Eigen::Matrix3d covariance_sph;
            covariance_sph          <<  sigma_sq_r, 0, 0,
                                        0, sigma_sq_theta, 0,
                                        0, 0, sigma_sq_phi;

            // std::cerr << "\nspherical Point covariance matrix: " << covariance_sph << std::flush;

            // calculate partial derivatives
            double dx_dr            =   cos(point.azimuth) * cos(elevation);
            double dx_dtheta        =   -point.range * sin(point.azimuth) * cos(elevation);
            double dx_dphi          =   -point.range * cos(point.azimuth) * sin(elevation);
            double dy_dr            =   sin(point.azimuth) * cos(elevation);
            double dy_dtheta        =   point.range * cos(point.azimuth) * cos(elevation);
            double dy_dphi          =   -point.range * sin(point.azimuth) * sin(elevation);
            double dz_dr            =   sin(elevation);
            double dz_dtheta        =   0.0;
            double dz_dphi          =   point.range * cos(elevation);

            Eigen::Matrix3d J;
            J                       <<  dx_dr, dx_dtheta, dx_dphi,
                                        dy_dr, dy_dtheta, dy_dphi, 
                                        dz_dr, dz_dtheta, dz_dphi;
          
            Eigen::Matrix3d covariance_cart;
            covariance_cart         =   J * covariance_sph * J.transpose();
            covariance_cart(0,2) = 0.0; covariance_cart(2,0) = 0.0;
            covariance_cart(1,2) = 0.0; covariance_cart(2,1) = 0.0;

            
            point.cov               =   covariance_cart;
        }
    }

    void compute_gaussian_pose(datastructures::Gaussian& gaussian) 
    {
        // perform eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(gaussian.cov);

        // check if solve was success
        if(es.info() != Eigen::Success)
        {
            std::cerr << "Unsuccessful eigen value decomposition" << std::flush;
        }
        
        // extracting eigen vectors
        // The columns of the matrix returned by eigenvectors() are the eigenvectors.
        // These eigenvectors form an orthonormal basis that represents the orientation
        // of the ellipsoid's principal axes in space.
 
        Eigen::Vector3d eigenvalues = es.eigenvalues();
        Eigen::Matrix3d eigenvectors = es.eigenvectors();
        std::vector<int> p(3);
        std::iota(p.begin(), p.end(), 0); // Fills p with 0, 1, 2

        // Sort indices based on the values of the eigenvalues in descending order
        // Eigenvalues are naturally returned in ascending order by SelfAdjointEigenSolver
        std::sort(p.begin(), p.end(), [&](int i, int j)
        {
            return eigenvalues(i) > eigenvalues(j); // Sort descending
        });

        Eigen::Matrix3d rotmat;
        rotmat.col(0) = eigenvectors.col(p[0]); // Major axis (largest eigenvalue)
        rotmat.col(1) = eigenvectors.col(p[1]); // Middle axis
        rotmat.col(2) = eigenvectors.col(p[2]); // Minor axis (smallest eigenvalue)


        if(rotmat.determinant() < 0.0) // checking for improper rotation matrix
        {
            rotmat.col(2) *= -1.0;
        }

        // Eigenvalue/Eigenvector order:.
        // The columns of the rotationMatrix correspond to the eigenvalues in the same order.

        // constructing pose
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();

        pose.block<3, 3>(0, 0) = rotmat;
        pose.block<3, 1>(0, 3) = gaussian.centroid;
        gaussian.pose = pose;

        // including eigen values into the gaussian
        Eigen::Vector3d sorted_eigenvalues;
        sorted_eigenvalues(0) = eigenvalues(p[0]);
        sorted_eigenvalues(1) = eigenvalues(p[1]);
        sorted_eigenvalues(2) = eigenvalues(p[2]);
        gaussian.eigenvalues = sorted_eigenvalues;
        
    }

    datastructures::GaussianMixture create_gaussian_mixture(datastructures::ClusterMap& temp_cluster_data)
    {
        bool ensure_2d  =   true;
        datastructures::GaussianMixture gaussian_mixture;

        for(auto& entry : temp_cluster_data)
        {
            int cluster_id                              =   entry.first;
            datastructures::ClusterData cluster_data    =   entry.second;
            // std::cerr << "\nsample data (id, range, azimuth): (" << cluster_id << "," << cluster_data.points[1].range << ", " << cluster_data.points[1].azimuth << ")" << std::flush;
            if(cluster_id == -2)
            {
                continue;
            }

            datastructures::Gaussian gaussian;
            gaussian.id         =   cluster_id;
            gaussian.velocity   =   0.0;
            gaussian.noise      =   0.0;
            gaussian.intensity  =   0.0;
            gaussian.weight     =   0.0;

            // calculate the covariance of the point measurements
            calculate_covariance(entry.second);
            calculate_covariance_2d(entry.second);

            // calculate weights and centroid of cluster
            double rcs_sum      =   0.0;

            for (const datastructures::Point& point : entry.second.points)
            {                
                rcs_sum         +=  point.rcs;
            }

            gaussian.centroid   =   Eigen::Vector3d::Zero();
            Eigen::Vector2d centroid_2d =   Eigen::Vector2d::Zero();
            Eigen::Matrix2d weighted_point_cov_sum_2d = Eigen::Matrix2d::Zero();
            if(entry.second.points.empty() || rcs_sum == 0.0)
            {
                std::cerr << "Error: empty cluster or zero rcs sum skipping cluster: " << cluster_id << std::flush;
                continue;
            }

            // calculate weighted centroid
            Eigen::Matrix3d weighted_point_cov_sum = Eigen::Matrix3d::Zero();
            for (const datastructures::Point& point : entry.second.points)
            {
                // Use the point's rcs relative to the total rcs as weight
                double weight       =   point.rcs / rcs_sum; 
                Eigen::Vector3d position;
                position            <<  point.x, point.y, 0.0;
                gaussian.centroid   +=  (weight * position);

                // std::cerr << "\ngaussian centroid of id: " << gaussian.id << " : (" << gaussian.centroid << ")" << std::flush;
                // Calculating 2D weighted covariance
                Eigen::Vector2d position_2d;
                position_2d         <<  point.x, point.y;
                centroid_2d         +=  (weight * position_2d);
                weighted_point_cov_sum_2d  += (weight * point.cov_2d);
                
                // Accumulate unweighted averages for velocity, rcs, noise
                gaussian.velocity       += point.vel;
                gaussian.intensity      += point.rcs;
                gaussian.noise          += 0.0; 
                weighted_point_cov_sum  += (weight * point.cov);
            }

            double num_points           = static_cast<double>(entry.second.points.size());
            gaussian.velocity           /= num_points;
            gaussian.intensity          /= num_points;
            gaussian.noise              /= num_points;
            // The Gaussian weight is average rcs
            gaussian.weight = rcs_sum / num_points;  

            // calculate the final weighted covariance
            Eigen::Matrix3d mean_deviation_cov = Eigen::Matrix3d::Zero();
            Eigen::Matrix2d mean_deviation_cov_2d=  Eigen::Matrix2d::Zero();
            for (const datastructures::Point& point : entry.second.points)
            {
                Eigen::Vector3d position;
                position                        <<  point.x, point.y, 0.0;
                Eigen::Vector3d diff            = (position - gaussian.centroid); // Deviation from the mean

                Eigen::Matrix3d outer_prod = diff * diff.transpose();

                // Get the weight for this point (relative rcs)
                double weight  = (point.rcs / rcs_sum);

                // std::cerr << "\nDEBUG: Point in cluster " << cluster_id << " (x,y,z,rcs): ("
                //           << point.x << ", " << point.y << ", " << point.z << ", " << point.rcs << ")" << std::endl;
                // std::cerr << "\nDEBUG: Centroid (x,y,z): ("
                //           << gaussian.centroid.x() << ", " << gaussian.centroid.y() << ", " << gaussian.centroid.z() << ")" << std::endl;
                // std::cerr << "\nDEBUG: Diff (dx,dy,dz): " << diff.transpose() << std::endl;
                // std::cerr << "\nDEBUG: Weight: " << weight << std::endl;
                // std::cerr << "\nDEBUG: Outer Product (diff * diff.transpose()):\n" << outer_prod << std::endl;
                // std::cerr << "\nDEBUG: Current mean_deviation_cov (BEFORE adding this point):\n" << mean_deviation_cov << std::endl;

                // Add weighted outer product to the covariance accumulator
                mean_deviation_cov += (weight * outer_prod);
            }

            gaussian.cov                   =    mean_deviation_cov + weighted_point_cov_sum;
            gaussian.cov_2d                =    mean_deviation_cov.block<2,2>(0, 0) + weighted_point_cov_sum_2d;
            compute_gaussian_pose(gaussian);        
            gaussian_mixture[cluster_id]    =   gaussian;
            
        }

        // normalize the weights of the gaussians
        double max_weight = 0.0, min_weight = INFINITY;
        for (auto entry : gaussian_mixture)
        {
            if(entry.second.weight < min_weight)
            {
                min_weight = entry.second.weight;
            }
            if(entry.second.weight > max_weight)
            {
                max_weight = entry.second.weight;
            }
        }
        for (auto& entry : gaussian_mixture)
        {
            entry.second.weight = (entry.second.weight - min_weight) / (max_weight - min_weight); 
        }
        
        return gaussian_mixture;
    }

    datastructures::Ellipse2D calculate_ellipse(const Eigen::Matrix2d& covariance)
    {

        datastructures::Ellipse2D ellipse_params;
        Eigen::EigenSolver<Eigen::Matrix2d> eigensolver(covariance);
        Eigen::Vector2d eigenvalues = eigensolver.eigenvalues().real();
        Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors().real();

        if (eigenvalues(0) < eigenvalues(1)) 
        {
            std::swap(eigenvalues(0), eigenvalues(1));
            eigenvectors.col(0).swap(eigenvectors.col(1));
        }

        // calculating axis lengths
        double axis1 = std::sqrt(eigenvalues(0));;
        double axis2 = std::sqrt(eigenvalues(1));
        double angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0));
        angle       =   (angle * 180) / M_PI;
        
        ellipse_params.axis_1 = axis1;
        ellipse_params.axis_2 = axis2;
        ellipse_params.orientation = angle;

        return ellipse_params;
    }


    void create_gaussian_ellipsoid_markers( const datastructures::GaussianMixture& gaussian_mixture, 
                                            visualization_msgs::msg::MarkerArray& marker_array, 
                                            datastructures::MarkerHistory& marker_queue,
                                            const std_msgs::msg::Header& header,
                                            float scale_factor, 
                                            float min_scale    
                                        )
    {
        visualization_msgs::msg::MarkerArray current_markers;
        // Define a namespace for these markers
        std::string marker_namespace = "gaussian_mixture_ellipsoids";

        // Define default color 
        std_msgs::msg::ColorRGBA default_color;
        default_color.r = 0.0f;
        default_color.g = 1.0f;
        default_color.b = 0.0f;
        default_color.a = 0.5f; 

        

        // Iterate through each Gaussian in the mixture
        for (const auto& pair : gaussian_mixture)
        {
            int gaussian_id = pair.first;
            const datastructures::Gaussian& gaussian = pair.second;


            if(gaussian_id != -2)
            {
                // Create a new Marker
                visualization_msgs::msg::Marker marker;

                // --- Set basic marker properties ---
                marker.header = header;
                marker.ns = marker_namespace;
                marker.id = gaussian_id; 
                marker.type = visualization_msgs::msg::Marker::SPHERE; 
                marker.action = visualization_msgs::msg::Marker::ADD; 

                // --- Set Pose (Position and Orientation) ---
                marker.pose.position.x = gaussian.pose(0, 3);
                marker.pose.position.y = gaussian.pose(1, 3);
                marker.pose.position.z = gaussian.pose(2, 3);

                Eigen::Matrix3d rotation_matrix = gaussian.pose.block<3, 3>(0, 0);
                
                Eigen::Quaterniond quaternion(rotation_matrix);
                quaternion.normalize(); 

                marker.pose.orientation.x = quaternion.x();
                marker.pose.orientation.y = quaternion.y();
                marker.pose.orientation.z = quaternion.z();
                marker.pose.orientation.w = quaternion.w();

                // --- Set Scale ---
                int z_axis_eigenvector_idx = -1;
                double max_z_component_magnitude = -1.0;

                for (int i = 0; i < 3; ++i) 
                {
                    double current_z_component_magnitude = std::abs(rotation_matrix.col(i).z());
                    
                    if (current_z_component_magnitude > (1.0 - 1e-6) ) 
                    { 
                        z_axis_eigenvector_idx = i;
                        break;
                    }
                    
                    if (current_z_component_magnitude > max_z_component_magnitude) 
                    {
                        max_z_component_magnitude = current_z_component_magnitude;
                        z_axis_eigenvector_idx = i;
                    }
                }

                if (z_axis_eigenvector_idx == -1) 
                {
                    std::cerr << "\nCould not find Z-aligned eigenvector for gaussian: "<< gaussian.id << "Defaulting to smallest eigenvalue for thickness." << std::flush;
                    z_axis_eigenvector_idx = 0; 
                }
                
                marker.scale.z = 2.0 * scale_factor * std::sqrt(std::max(0.0, gaussian.eigenvalues(z_axis_eigenvector_idx)));
                
                std::vector<double> xy_eigenvalues_for_marker_scales;
                for (int i = 0; i < 3; ++i) 
                {
                    if (i != z_axis_eigenvector_idx) 
                    {
                        xy_eigenvalues_for_marker_scales.push_back(gaussian.eigenvalues(i));
                    }
                }
                // Ensure we have exactly two XY eigenvalues
                if (xy_eigenvalues_for_marker_scales.size() == 2) 
                {
                    marker.scale.x = 2.0 * scale_factor * std::sqrt(std::max(0.0, xy_eigenvalues_for_marker_scales[0]));
                    marker.scale.y = 2.0 * scale_factor * std::sqrt(std::max(0.0, xy_eigenvalues_for_marker_scales[1]));
                }
                else 
                {
                 marker.scale.x = min_scale; 
                 marker.scale.y = min_scale;
                }
                marker.scale.x = std::max(marker.scale.x, static_cast<double>(min_scale));
                marker.scale.y = std::max(marker.scale.y, static_cast<double>(min_scale));
                marker.scale.z = std::max(marker.scale.z, static_cast<double>(min_scale)); 
                // --- Set Color ---
                marker.color = generateColorFromGaussianID(gaussian_id, 0.7f); 

                // --- Set Lifetime ---
                // 0 indicates infinite lifetime (marker will persist until deleted or replaced)
                marker.lifetime.sec = 0.0;
                marker.lifetime.nanosec = 1;

                // Frame locked determines if the marker moves with the frame or stays fixed in the world
                marker.frame_locked = false; 

                // Add the populated marker to the array
                current_markers.markers.push_back(marker);
            }    
        }

        if (marker_queue.size() >= 20) 
        {
            marker_queue.pop_front();
        }
        marker_queue.push_back(current_markers);

        marker_array.markers.clear();

        if(!marker_queue.empty())
        {
            marker_array.markers    =   marker_queue.back().markers;
        }

        int history_marker_id_counter = 0;
        for(size_t i = 0; i < marker_queue.size() - 1; ++i)
        {
            const auto& historical_array = marker_queue[i];
            for (const auto& original_marker : historical_array.markers)
            {
                visualization_msgs::msg::Marker history_marker = original_marker;
                history_marker.ns = "history";
                history_marker.color = generateColorFromGaussianID(-2, 0.2);
                history_marker.id = history_marker_id_counter++;
                marker_array.markers.push_back(history_marker);
            }
        }  
    }

    // void update_unique_gaussians(datastructures::GaussianMixture& unique_gaussians, datastructures::GaussianMixture& new_gaussians)
    // {
        
    // }
}


