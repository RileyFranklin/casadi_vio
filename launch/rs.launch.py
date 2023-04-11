from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = False
    logger = LaunchConfiguration("log_level")
    return LaunchDescription([
        # ExecuteProcess(cmd=['ros2', 'bag', 'record',
        #    '/camera/color/image_raw',
        #    '/camera/color/camera_info',
        #    '/camera/depth/color/points',
        # ]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([get_package_share_directory('realsense2_camera') ,'/launch/rs_launch.py']),
            launch_arguments={
                'pointcloud.enable': 'true',
                'align_depth.enable': 'true',
                'depth_module.profile': '848,480,60',
                'rgb_camera.profile': '848,480,60',
        		'pointcloud.ordered_pc': 'true',
                }.items(),
        ),
        Node(
           package='tf2_ros',
           output='log',
           executable='static_transform_publisher',
           parameters=[
             {'use_sim_time': use_sim_time}
           ],
           arguments=[
               "0", "0", "0", "0", "0", "0",
               "map", "test_frame"]
        ),
        Node(
           package='tf2_ros',
           output='log',
           executable='static_transform_publisher',
           parameters=[
             {'use_sim_time': use_sim_time}
           ],
           arguments=[
               "0", "0", "0", "-0.5", "0.5", "-0.5", "0.5",
               "camera_frame", "camera_color_optical_frame"]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=[
                '-d',
                get_package_share_directory('casadi_vio') + '/config/realsense_vio.rviz',
                '--ros-args', '--log-level', logger],
            parameters=[{'use_sim_time': use_sim_time}],
            on_exit=Shutdown(),
        ),
        Node(
            package='casadi_vio',
            executable='depth_from_pc.py',
        ),
        #Node(package='casadi_vio', executable='feature_points.py', output='screen')
    ])
