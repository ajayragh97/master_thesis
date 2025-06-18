#ifndef HELPERFUNCTIONS_HPP
#define HELPERFUNCTIONS_HPP
#include "datastructure.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/header.hpp> 
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace helperfunctions
{
    datastructures::ClusterMap create_dataframe(std::vector<datastructures::Point>& points);
    datastructures::GaussianMixture create_gaussian_mixture(datastructures::ClusterMap& cluster_data);
    void create_gaussian_ellipsoid_markers(const datastructures::GaussianMixture& gaussian_mixture, 
                                        visualization_msgs::msg::MarkerArray& marker_array, 
                                        datastructures::MarkerHistory& marker_queue,
                                        const std_msgs::msg::Header& header,
                                        float scale_factor = 2.0,
                                        float min_scale = 0.01);    
    std_msgs::msg::ColorRGBA generateColorFromGaussianID(int gaussian_id, float alpha);  
    // void update_unique_gaussians(datastructures::GaussianMixture& unique_gaussians, datastructures::GaussianMixture& new_gaussians);                               
}

#endif