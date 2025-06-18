#include <chrono>
#include <cmath>
#include <Eigen/Geometry>
#include <cstring>
#include <random>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "../include/master_thesis/datastructure.hpp"
#include "../include/master_thesis/dbscan.hpp"
#include "../include/master_thesis/helperfunctions.hpp"


class   ClustererNode :   public  rclcpp::Node
{
    public:
        ClustererNode(double eps, int min_pts, int count) : Node("target_tracking_node"), eps_(eps), min_pts_(min_pts), count_(count)
        {
            pcl_sub          =   this -> create_subscription<sensor_msgs::msg::PointCloud2>
                                ("/radarpcl_gc", rclcpp::SystemDefaultsQoS(), std::bind(&ClustererNode::pointcloud_callback, this, std::placeholders::_1));
            
            odom_sub        =   this -> create_subscription<nav_msgs::msg::Odometry>
                                ("/odom", rclcpp::SystemDefaultsQoS(), std::bind(&ClustererNode::odometry_callback, this, std::placeholders::_1));
            
            dbscan          =   std::make_shared<clustering::DBSCAN_cluster>(eps, min_pts);  
            
            ellipse_pub     =   this -> create_publisher<visualization_msgs::msg::MarkerArray>("ellipse", rclcpp::SystemDefaultsQoS()); 
            
            pointcloud_pub  =   this -> create_publisher<sensor_msgs::msg::PointCloud2>("clustered_cloud", rclcpp::SystemDefaultsQoS()); 

            odom_pub        =   this -> create_publisher<nav_msgs::msg::Odometry>("odom_corrected", rclcpp::SystemDefaultsQoS());
        }
    
    private:

        double eps_;
        int min_pts_;
        int count_;
        
        std::vector<datastructures::Point> radar_data;
        Eigen::Vector3f origin;
        int msg_count   =   0;
        bool origin_set =   false;
        datastructures::GaussianMixture unique_gaussians;
        datastructures::MarkerHistory marker_queue;

        void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
        {
            // geometry_msgs::msg::TransformStamped t;
            if(!origin_set)
            {
                origin      <<  msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
                origin_set  =   true;
            }

            if(origin_set)
            {
                nav_msgs::msg::Odometry odom_msg;
                odom_msg.child_frame_id         =   "base_link";
                odom_msg.header.frame_id        =   "map";
                odom_msg.header.stamp           =   this->now();
                odom_msg.pose.covariance        =   msg->pose.covariance; 
                odom_msg.pose.pose.orientation  =   msg->pose.pose.orientation;
                odom_msg.pose.pose.position.x   =   msg->pose.pose.position.x - origin(0);
                odom_msg.pose.pose.position.y   =   msg->pose.pose.position.y - origin(1);
                odom_msg.pose.pose.position.z   =   msg->pose.pose.position.z - origin(2);
                odom_msg.twist                  =   msg->twist;

                odom_pub->publish(odom_msg);
            }
        }

        void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {

            if(!origin_set)
            {
                return;
            }

            msg_count += 1;
            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
            sensor_msgs::PointCloud2ConstIterator<float> iter_range(*msg, "range");
            sensor_msgs::PointCloud2ConstIterator<float> iter_azimuth(*msg, "azimuth");
            sensor_msgs::PointCloud2ConstIterator<float> iter_velocity(*msg, "vr");
            sensor_msgs::PointCloud2ConstIterator<float> iter_velocity_compensated(*msg, "vr_c");
            sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "rcs");
            sensor_msgs::PointCloud2ConstIterator<float> iter_label(*msg, "label");

            for (size_t i = 0; i < msg->width * msg->height; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_range, ++iter_azimuth, ++iter_velocity, ++iter_velocity_compensated, ++iter_intensity, ++iter_label) {

                double   x           =   static_cast<double>(*iter_x);
                double   y           =   static_cast<double>(*iter_y);
                double   z           =   static_cast<double>(*iter_z);
                double   range       =   static_cast<double>(*iter_range);
                double   azimuth     =   static_cast<double>(*iter_azimuth);
                double   velocity    =   static_cast<double>(*iter_velocity);
                double   velocity_c  =   static_cast<double>(*iter_velocity_compensated);
                double   intensity   =   static_cast<double>(*iter_intensity);
                uint8_t label        =   *iter_label;
                
                // transforming cordinates relative to first odometry position
                x   -=  origin(0);
                y   -=  origin(1);
                z   -=  origin(2);

                // rcs values are in dBsm, so it can be negative, this will cause issues in weight calculation
                // converting from logarithmic to linear scale
                double rcs_linear   = std::max(1e-6, std::pow(10.0, intensity / 10.0)); 
                datastructures::Point radar_point;
                radar_point.x           =   x;
                radar_point.y           =   y;
                radar_point.z           =   z;
                radar_point.azimuth     =   azimuth;
                radar_point.range       =   range;
                radar_point.vel         =   velocity;
                radar_point.rcs         =   rcs_linear;   
                
                radar_data.push_back(radar_point);

            }

            if (msg_count > count_)
            {
                // perform dbscan filtering 
                auto start      =   std::chrono::high_resolution_clock::now();
                dbscan          ->  fit(radar_data);
                auto end        =   std::chrono::high_resolution_clock::now();
                auto duration   = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // std::cout << "\ndata after dbscan (range, azimuth): (" << radar_data[0].range << ", " << radar_data[0].azimuth  << ")" << std::flush;
                // std::cout << "dbscan time: " << duration.count() << " microseconds\n";

                // Arranging data into defined datastructure
                datastructures::ClusterMap cluster_data;
                start           = std::chrono::high_resolution_clock::now();

                cluster_data    = helperfunctions::create_dataframe(radar_data);
                // std::cout << "\n data after dataframe creation (range, azimuth): (" << cluster_data[0].points[0].range << ", " << cluster_data[0].points[0].azimuth << ")" << std::flush;
                end             = std::chrono::high_resolution_clock::now();
                duration        = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                // std::cout << "dataframe creation time: " << duration.count() << " microseconds\n";
                
                // creating the gaussian mixture from the clusters
                datastructures::GaussianMixture gaussian_mixture;
                gaussian_mixture    =   helperfunctions::create_gaussian_mixture(cluster_data);
                rclcpp::Time stamp  =   msg->header.stamp;
                int total_points    =   radar_data.size();

                // identifying unique clusters
                if(unique_gaussians.empty())
                {
                    unique_gaussians    =   gaussian_mixture;
                }
                else
                {
                    // helperfunctions::update_unique_gaussians(unique_gaussians, gaussian_mixture);
                }
                publish_ellipsoid(gaussian_mixture, stamp);
                publish_clustered_pointcloud(cluster_data, stamp);

                radar_data.clear();
                msg_count = 0;
            }
        }

        void publish_ellipsoid(const datastructures::GaussianMixture& gaussian_mixture, const rclcpp::Time& stamp)
        {
            visualization_msgs::msg::MarkerArray marker_array_msg;
            std_msgs::msg::Header header;
            header.stamp = this->now(); 
            header.frame_id = "map";
            float scale_factor = 2.0;
            float min_diameter = 0.1;

            helperfunctions::create_gaussian_ellipsoid_markers(gaussian_mixture, marker_array_msg, marker_queue, header, scale_factor, min_diameter);
            ellipse_pub->publish(marker_array_msg);  
        }

        void publish_clustered_pointcloud(const datastructures::ClusterMap& clusters, const rclcpp::Time& stamp)
        {
            int num_points = 0;
            for (auto& entry    :   clusters)
            {
                num_points += static_cast<int>(entry.second.points.size());
            }

            sensor_msgs::msg::PointCloud2 cloud_msg;
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "map";

            

            // Define additional fields for RGB
            int rgb_offset = sizeof(float) * 3; // Offset after xyz and cluster_id

            // Creating fileds
            cloud_msg.height = 1;
            cloud_msg.width = num_points;
            cloud_msg.is_bigendian = false;
            cloud_msg.is_dense = true;
            cloud_msg.point_step = sizeof(float) * 3 + sizeof(uint32_t);
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

            // Defining the fields (x, y, z, cluster_id)
            sensor_msgs::msg::PointField field;
            field.datatype = sensor_msgs::msg::PointField::FLOAT32;
            field.count = 1;

            field.name = "x";
            field.offset = 0;
            cloud_msg.fields.push_back(field);

            field.name = "y";
            field.offset = 4;
            cloud_msg.fields.push_back(field);

            field.name = "z";
            field.offset = 8;
            cloud_msg.fields.push_back(field);

            // RGB field definition
            sensor_msgs::msg::PointField rgb_field;
            rgb_field.datatype = sensor_msgs::msg::PointField::UINT32; // 32-bit unsigned integer for RGB
            rgb_field.count = 1; // Now only 1 count, as it's a single 32-bit value
            rgb_field.name = "rgb";
            rgb_field.offset = rgb_offset;
            cloud_msg.fields.push_back(rgb_field);

            // Allocate data
            cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

            size_t current_byte_offset = 0;
            for(const auto& entry : clusters)
            {
                int cluster_id = entry.first;
                datastructures::ClusterData data = entry.second;
                std::vector<datastructures::Point> point_data = data.points;

                std_msgs::msg::ColorRGBA color_rgba;
                color_rgba          =   helperfunctions::generateColorFromGaussianID(cluster_id, 1.0);
                uint8_t r_u8        =   static_cast<uint8_t>(color_rgba.r * 255.0f);
                uint8_t g_u8        =   static_cast<uint8_t>(color_rgba.g * 255.0f);
                uint8_t b_u8        =   static_cast<uint8_t>(color_rgba.b * 255.0f);
                uint8_t a_u8        =   static_cast<uint8_t>(color_rgba.a * 255.0f);

                uint32_t rgb_color =    (static_cast<uint32_t>(a_u8) << 24) |
                                        (static_cast<uint32_t>(r_u8) << 16) |
                                        (static_cast<uint32_t>(g_u8) << 8)  |
                                        (static_cast<uint32_t>(b_u8));

                for(size_t j = 0; j < point_data.size(); ++j)
                {                       
                    uint8_t *data_ptr = &cloud_msg.data[current_byte_offset];
                    float px = static_cast<float>(point_data[j].x);
                    float py = static_cast<float>(point_data[j].y);
                    float pz = static_cast<float>(point_data[j].z);                
                    memcpy(data_ptr + 0, &px, sizeof(float));
                    memcpy(data_ptr + 4, &py, sizeof(float));
                    memcpy(data_ptr + 8, &pz, sizeof(float));
                    memcpy(data_ptr + rgb_offset, &rgb_color, sizeof(uint32_t));
                    current_byte_offset += cloud_msg.point_step;                    
                } 


            }
            pointcloud_pub->publish(cloud_msg);

            
        }


        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_sub;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
        std::shared_ptr<clustering::DBSCAN_cluster> dbscan;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ellipse_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    double eps = 1;
    int min_pts = 2;
    int count = 1;
    
    rclcpp::spin(std::make_shared<ClustererNode>(eps, min_pts, count));
    rclcpp::shutdown();
    return 0;
}