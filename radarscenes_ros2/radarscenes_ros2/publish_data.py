import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TransformStamped
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import numpy as np
import h5py
import json
import time
import csv



class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('pointcloud_publisher')
        self.publisher_cc = self.create_publisher(PointCloud2, 'radarpcl_cc', 10)
        self.publisher_gc = self.create_publisher(PointCloud2, 'radarpcl_gc', 10)
        self.publisher_sc = self.create_publisher(PointCloud2, 'radarpcl_sc', 10)
        self.publisher_agg= self.create_publisher(PointCloud2, 'radarpcl_agg', 10)
        self.publisher_odom = self.create_publisher(Odometry, 'odom', 10)
        self.publisher_tf_dynamic = TransformBroadcaster(self)
        self.publisher_tf_static = StaticTransformBroadcaster(self)
        self.declare_parameter('seq', 145)
                
        self.dataset_path = "/home/ajay/work/Fraunhofer/dataset/RadarScenes/data"
        self.seq = self.get_parameter('seq').get_parameter_value().integer_value
        self.data_file = "{}/sequence_{}/radar_data.h5".format(self.dataset_path, self.seq)
        self.meta_file = "{}/sequence_{}/scenes.json".format(self.dataset_path, self.seq)
        self.sensor_file = "{}/sensors.json".format(self.dataset_path)
        self.csv_path = "{}/sequence_{}/odom.csv".format(self.dataset_path, self.seq)
        self.sensor_ids = {1, 2, 3, 4}
        
        with h5py.File(self.data_file, 'r') as file:

            self.radar_data = file['radar_data'][:]
            self.odom_data = file['odometry'][:]

            print(f"radar data type: {self.radar_data.dtype}, odom data type: {self.odom_data.dtype}")
        
        # saving data to csv
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(self.odom_data)

        with open(self.meta_file, 'r') as file:
            self.meta_data = json.load(file)
        
        with open(self.sensor_file, 'r') as file:
            self.sensor_data = json.load(file)

        self.current_pcl_idx = 0
        
        self.pcl_scenes = self.meta_data['scenes']
        self.pcl_timestamps = list(self.pcl_scenes.keys())

        self.sensor_scan_buffer = {}

        self.publish_static_transform()
        print(f'publishing sequence {self.seq}')
        

        
    def publish_static_transform(self):
        static_transforms = []
        for sensor in self.sensor_data.keys():
            radar_id = self.sensor_data[sensor]['id']
            x = self.sensor_data[sensor]['x']
            y = self.sensor_data[sensor]['y']
            yaw = self.sensor_data[sensor]['yaw']

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = 'base_link'
            tf_msg.child_frame_id = f'radar_{radar_id}'

            tf_msg.transform.translation.x = float(x)
            tf_msg.transform.translation.y = float(y)
            tf_msg.transform.translation.z = 0.0

            q = quaternion_from_euler(0, 0, float(yaw))
            tf_msg.transform.rotation = Quaternion(x = q[0], y = q[1], z = q[2], w = q[3])

            static_transforms.append(tf_msg)

        self.publisher_tf_static.sendTransform(static_transforms)
        self.get_logger().info(f"publishing static transforms")


    def create_pcl_msg(self):

        if(self.current_pcl_idx >= self.radar_data.shape[0]):
            return
        

        stamp = self.pcl_timestamps[self.current_pcl_idx]
        sensor_id = self.pcl_scenes[stamp]['sensor_id']
        start_idx = self.pcl_scenes[stamp]['radar_indices'][0]
        end_idx = self.pcl_scenes[stamp]['radar_indices'][1]
        
        cc_data = []
        gc_data = [] 
        sc_data = []

        for i in range(start_idx, end_idx):
            
            range_sc = self.radar_data[i]['range_sc']
            azimuth_sc = self.radar_data[i]['azimuth_sc']
            rcs = self.radar_data[i]['rcs']
            vr = self.radar_data[i]['vr']
            vr_c = self.radar_data[i]['vr_compensated']
            x_cc = self.radar_data[i]['x_cc']
            y_cc = self.radar_data[i]['y_cc']
            z_cc = 0.0
            x_gc = np.float32(self.radar_data[i]['x_seq'])
            y_gc = np.float32(self.radar_data[i]['y_seq'])
            z_gc = np.float32(0.0)
            label = np.float32(self.radar_data[i]['label_id'])

            cc = [x_cc, y_cc, z_cc, range_sc, azimuth_sc, vr, vr_c, rcs, label]
            gc = [x_gc, y_gc, z_gc, range_sc, azimuth_sc, vr, vr_c, rcs, label]
            sc = [range_sc, azimuth_sc, rcs, vr, label]

            cc_data.append(cc)
            gc_data.append(gc)
            sc_data.append(sc)
        
        cc_data = np.asarray(cc_data)
        gc_data = np.asarray(gc_data)
        sc_data = np.asarray(sc_data)

        if sensor_id in self.sensor_ids:
            self.sensor_scan_buffer[sensor_id] = gc_data

            if self.sensor_ids.issubset(self.sensor_scan_buffer.keys()):
                all_points = list(self.sensor_scan_buffer.values())
                aggregated_data = np.vstack(all_points)

                last_scan_time_us = int(stamp)
                aggregated_stamp = Time(sec=last_scan_time_us // 1_000_000, nanosec=(last_scan_time_us % 1_000_000) * 1000)

                self.publish_agg_pointcloud_msg(aggregated_data, "map", aggregated_stamp)

                self.sensor_scan_buffer.clear()

        if self.current_pcl_idx < self.radar_data.shape[0] - 1:
            next_timestamp = self.pcl_timestamps[self.current_pcl_idx + 1]
            time_delay = int(next_timestamp) - int(stamp)
        else:
            time_delay = 0

        time.sleep(time_delay / 1_000_000) 

        self.publish_pointcloud(cc_data, "base_link", stamp)
        self.publish_pointcloud(sc_data, "radar_{}".format(sensor_id), stamp)
        self.publish_pointcloud(gc_data, "map", stamp)

        self.current_pcl_idx += 1

    def publish_agg_pointcloud_msg(self, data, frame_id, stamp_msg):
        cloud_msg = PointCloud2()
        cloud_msg.header.frame_id = frame_id
        cloud_msg.header.stamp = stamp_msg

        cloud_msg.height = 1
        cloud_msg.width = data.shape[0]
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = True

        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='azimuth', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr_c', offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='rcs', offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=32, datatype=PointField.FLOAT32, count=1)
                            ]
        cloud_msg.point_step = 9 * (np.dtype(np.float32).itemsize)
        cloud_msg.row_step = cloud_msg.point_step * data.shape[0]
        cloud_msg.data = data.tobytes()
        self.publisher_agg.publish(cloud_msg)


        
    def publish_pointcloud(self, data, frame_id, timestamp):

        # Define cloud properties
        cloud_msg = PointCloud2()
        timestamp = int(timestamp)
        timestamp_msg = Time()
        seconds = timestamp // 1_000_000  # Integer division for seconds
        nanoseconds = (timestamp % 1_000_000) * 1_000  # Remaining microseconds converted to nanoseconds
        timestamp_msg.sec = int(seconds)
        timestamp_msg.nanosec = int(nanoseconds)
        cloud_msg.header.stamp = timestamp_msg
        cloud_msg.header.frame_id = frame_id  # Set the frame ID    

        # Define point cloud fields (x, y, vr, vr_c, rcs, label)

        cloud_msg.height = 1
        cloud_msg.width = data.shape[0]
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = True
        
        if(data.shape[1] == 9):
            cloud_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='range', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='azimuth', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='vr', offset=20, datatype=PointField.FLOAT32, count=1),
                PointField(name='vr_c', offset=24, datatype=PointField.FLOAT32, count=1),
                PointField(name='rcs', offset=28, datatype=PointField.FLOAT32, count=1),
                PointField(name='label', offset=32, datatype=PointField.FLOAT32, count=1)
            ]
            cloud_msg.point_step = 9 * (np.dtype(np.float32).itemsize)  # Each point has 9 fields x, y, z, range, azimuth, vr, vr_c, rcs, label (4 bytes each)
        
        elif(data.shape[1] == 5):
            cloud_msg.fields = [
            PointField(name='range', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='azimuth', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rcs', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.FLOAT32, count=1)
            ]
            cloud_msg.point_step = 5 * (np.dtype(np.float32).itemsize)
        
        cloud_msg.row_step = cloud_msg.point_step * data.shape[0]

        cloud_msg.data = data.tobytes()  # Convert to binary format
        # Publish the messages
        
        odom_msg, tf_msg = self.create_odom_msg(timestamp)

        if(tf_msg == None):
            pass
        else:
            self.publisher_odom.publish(odom_msg)
            # self.publisher_tf_dynamic.sendTransform(tf_msg)

        if(frame_id == "map"):
            self.publisher_gc.publish(cloud_msg)
        elif(frame_id == "base_link"):
            self.publisher_cc.publish(cloud_msg)
        else:
            self.publisher_sc.publish(cloud_msg)
        
         
        

    def create_odom_msg(self, pcl_stamp):
       
        timestamp_diff = np.abs(pcl_stamp - self.odom_data['timestamp'])
        lowest_time_diff = timestamp_diff.min()
        if (lowest_time_diff < 1000): # timestamp in microseconds, so check for odom data within 10 milliseconds
            
            odom_idx = np.argmin(timestamp_diff)
            odom_msg = Odometry()
            tf_msg = TransformStamped()
            

            odom_timestamp = self.odom_data[odom_idx][0]
            timestamp_msg = Time()
            seconds = odom_timestamp // 1_000_000  # Integer division for seconds
            nanoseconds = (odom_timestamp % 1_000_000) * 1_000  # Remaining microseconds converted to nanoseconds
            timestamp_msg.sec = int(seconds)
            timestamp_msg.nanosec = int(nanoseconds)
            odom_msg.header.stamp = timestamp_msg
            tf_msg.header.stamp = timestamp_msg

            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'
            tf_msg.header.frame_id = 'map'
            tf_msg.child_frame_id = 'base_link'

            odom_msg.pose.pose.position.x = float(self.odom_data[odom_idx]['x_seq'])
            odom_msg.pose.pose.position.y = float(self.odom_data[odom_idx]['y_seq'])
            odom_msg.pose.pose.position.z = 0.0

            tf_msg.transform.translation.x = float(self.odom_data[odom_idx]['x_seq'])
            tf_msg.transform.translation.y = float(self.odom_data[odom_idx]['y_seq'])
            tf_msg.transform.translation.z = 0.0

            yaw_gc = float(self.odom_data[odom_idx]['yaw_seq'])
            q = quaternion_from_euler(0, 0, yaw_gc)
            
            odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            tf_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            return odom_msg, tf_msg
        
        else:
            return None, None



        

def main(args=None):
    rclpy.init(args=args)
    pointcloud_publisher = PointCloudPublisher()

    try:
        while rclpy.ok():
            pointcloud_publisher.create_pcl_msg()
            rclpy.spin_once(pointcloud_publisher, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass


    # Shutdown
    pointcloud_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
