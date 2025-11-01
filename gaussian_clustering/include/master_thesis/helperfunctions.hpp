#ifndef HELPERFUNCTIONS_HPP
#define HELPERFUNCTIONS_HPP
#include "datastructure.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp> 
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include "nav_msgs/msg/odometry.hpp"


namespace helperfunctions
{
    datastructures::ClusterMap create_dataframe(std::vector<datastructures::Point>& points);
    datastructures::GaussianMixture create_gaussian_mixture(datastructures::ClusterMap& cluster_data,
                                                            const double& angle_max,
                                                            const double& angle_min,
                                                            const double& angle_var_max,
                                                            const double& angle_var_min,
                                                            const double& range_var,
                                                            const double& vel_thresh);
    void create_gaussian_ellipsoid_markers(const datastructures::GaussianMixture& gaussian_mixture, 
                                        const nav_msgs::msg::Odometry& odom_msg,
                                        visualization_msgs::msg::MarkerArray& marker_array, 
                                        datastructures::MarkerHistory& marker_queue,
                                        const std_msgs::msg::Header& header,
                                        const float& scale_factor,
                                        const float& min_scale,
                                        const float& alpha,
                                        const float& history_alpha,
                                        const int& history);    
    std_msgs::msg::ColorRGBA generateColorFromGaussianID(int gaussian_id, float alpha);  
    void create_occ_grid(const datastructures::GaussianMixture& gaussian_mixture,
                         const nav_msgs::msg::Odometry& odom_msg,
                         std::vector<std::vector<double>>& occ_grid,
                         const int& grid_rows, 
                         const int& grid_cols,
                         const double& cell_size,
                         const double& grid_offset);
}

#endif