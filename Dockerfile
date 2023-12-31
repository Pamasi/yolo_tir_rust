# This is an auto generated Dockerfile for ros:perception
# generated from docker_images_ros2/create_ros_image.Dockerfile.em
ARG ROS_DISTRO=humble
FROM osrf/ros:humble-desktop-full
ARG DEBIAN_FRONTEND=noninteractive

MAINTAINER pamasi
# install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libclang-dev \
    tmux \
    python3-pip \
    vim \
    xauth x11vnc xvfb tigervnc-viewer\
    && rm -rf /var/lib/apt/lists/*

# Install Rust and the cargo-ament-build plugin
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.74.0 -y
ENV PATH=/root/.cargo/bin:$PATH
RUN cargo install cargo-ament-build

RUN pip install --upgrade pytest colcon-package-selection

# Install the colcon-cargo and colcon-ros-cargo plugins
RUN pip install git+https://github.com/colcon/colcon-cargo.git git+https://github.com/colcon/colcon-ros-cargo.git

RUN pip install --upgrade pytest colcon-package-selection

# connect remotely
RUN useradd -ms /bin/bash user && yes password | passwd user
# create file in user home
RUN mkdir -p /home/user/workspace/src 

RUN git clone https://github.com/ros-perception/vision_msgs.git /home/user/workspace/src 
WORKDIR /home/user/workspace


