#!/bin/bash
# onnx library to shared library path
export LD_LIBRARY_PATH=/home/user/library/onnxruntime-linux-x64-1.13.1/lib:$LD_LIBRARY_PATH
# setup ros2 environment
. /opt/ros/humble/setup.sh
. install/local_setup.sh

#colcon build 