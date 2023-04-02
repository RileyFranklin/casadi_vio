from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    bag_file_arg = DeclareLaunchArgument('bag')
    bag_file = LaunchConfiguration('bag')
    return LaunchDescription([
        bag_file_arg,
        ExecuteProcess(cmd=['ros2', 'bag', 'play', bag_file, '--loop']),
        Node(package='casadi_vio', executable='feature_points.py', output='screen'),
    ])
