from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
       # ExecuteProcess(cmd=['ros2', 'bag', 'record',
       #     '/camera/color/image_rect_raw',
       #     '/camera/aligned_depth_to_color/image_raw'
       # ]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('realsense2_camera') ,'/launch/rs_launch.py']),
            launch_arguments={
            	#'pointcloud.enable': 'true',
                #'filters': 'pointcloud',
                'align_depth.enable': 'true',
                'depth_fps': '15.0',
                'depth_width': '640',
                'depth_height': '480',
                'color_fps': '15.0',
                'color_width': '640',
                'color_height': '480',
		        #'pointcloud.ordered_pc': 'true',
                #'camera_info_url': 'package://cuesr/config/depth_5_19_22.yaml',
                }.items(),
        ),
        Node(package='casadi_vio', executable='feature_points.py', output='screen')
    ])
