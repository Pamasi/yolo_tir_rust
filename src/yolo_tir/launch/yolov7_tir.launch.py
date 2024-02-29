import os.path  as osp
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import TextSubstitution, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def get_pkg_path(filename:str, pkg_name:str='yolo_tir'):
    return osp.join(
        get_package_share_directory(pkg_name), 
        filename
        )

def generate_launch_description():
    return LaunchDescription([
            # set to debug rust program
            SetEnvironmentVariable(name='RUST_BACKTRACE', value='full'),
            DeclareLaunchArgument(
                'log_level',
                default_value = TextSubstitution(text=str('WARN')),
                description='Logging level'
            ),
            Node(
                package="yolo_tir",
                executable="detection_pub",
                name="inference_node",
                output="screen",
                emulate_tty=True,
                arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
                # parse parameters from file for scalability
                parameters=[  get_pkg_path('param/net_config.yaml') ]
            )
    ])