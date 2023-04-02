from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=['ros2', 'bag', 'record',
            '/camera/color/image_raw',
            '/camera/aligned_depth_to_color/image_raw'
        ]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('realsense2_camera') ,'/launch/rs_launch.py']),
            launch_arguments={
                #'pointcloud.enable': 'true',
                'align_depth.enable': 'true',
                'depth_module.profile': '1280,720,30',
                'rgb_camera.profile': '1280,720,30',
		        #'ordered_pc': 'true',
                }.items(),
        ),
        Node(package='casadi_vio', executable='feature_points.py', output='screen')
    ])
