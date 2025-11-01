import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pkg_share_path = get_package_share_directory('master_thesis')

    config_filepath = os.path.join(pkg_share_path, 'config', 'config.yaml')

    return LaunchDescription([
        Node(
            package='master_thesis',
            executable='clustering_node',
            name='clustering_node',  
            parameters=[config_filepath]
        )
    ])

if __name__ == '__main__':
    pass