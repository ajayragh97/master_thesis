from setuptools import find_packages, setup

package_name = 'radarscenes_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ajayragh345@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_pointcloud = radarscenes_ros2.publish_data:main',
            'image_publisher = radarscenes_ros2.publish_images:main'
        ],
    },
)
