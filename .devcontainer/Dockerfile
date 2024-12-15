# Use NVIDIA's PyTorch image as the base
FROM dustynv/pytorch:2.1-r36.2.0

SHELL ["/bin/bash", "-c"]

# Set the timezone
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=humble
ENV ROS_DISTRO=${ROS_DISTRO}


RUN export PYTHON_VERSION=$(python3 --version | cut -d" " -f2 | cut -d. -f1-2) && \
    (sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/version.py && \
    sed -i 's/2.0.0.nv23.05/2.0.0/g' /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch-2.0.0.nv23.05.dist-info/METADATA) || \
    (apt update && apt install -y curl && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update && apt install -y ros-${ROS_DISTRO}-ros-base ros-dev-tools libboost-python-dev ros-${ROS_DISTRO}-rmw-cyclonedds-cpp ros-${ROS_DISTRO}-vision-msgs && \
    echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /root/.bashrc && \
    # echo export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp >> /root/.bashrc && \
    # apt install -y ros-${ROS_DISTRO}-image-common && \
    # export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
    rm -rf /var/lib/apt/lists/*)
    

RUN apt update && apt install -y ros-dev-tools && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-ament-cmake \
    libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*


# Set up ROS workspace
RUN mkdir -p /ros2_pkgs_ws/src && \
    cd /ros2_pkgs_ws && \
    git clone https://github.com/ros2/common_interfaces.git -b ${ROS_DISTRO} src/common_interfaces --depth 1 && \
    git clone https://github.com/ros-perception/vision_opencv.git -b ${ROS_DISTRO} src/vision_opencv --depth 1 && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --packages-skip-build-finished

    
# Source ROS setup
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc
RUN echo "source /ros2_pkgs_ws/install/local_setup.bash" >> /root/.bashrc


# RUN source /opt/ros/${ROS_DISTRO}/install/setup.bash || source /opt/ros/${ROS_DISTRO}/setup.bash && \
#     git clone https://github.com/ros2/common_interfaces.git -b ${ROS_DISTRO} /ros2_pkgs_ws/src/common_interfaces --depth 1 && \
#     git clone https://github.com/ros-perception/vision_opencv.git -b ${ROS_DISTRO} /ros2_pkgs_ws/src/vision_opencv --depth 1 && \
#     git clone https://github.com/mentee-robotics/aruco_ros.git -b v24.02.RC0 /ros2_pkgs_ws/src/aruco_ros --depth 1 && \
#     git clone https://github.com/ANYbotics/grid_map.git -b ${ROS_DISTRO} /tmp/grid_map --depth 1 && \
#     cp -r /tmp/grid_map/grid_map_msgs /ros2_pkgs_ws/src/ && \
#     cp -r /tmp/grid_map/grid_map_cmake_helpers /ros2_pkgs_ws/src/ && \
#     cd /ros2_pkgs_ws/ && \
#     colcon build --packages-skip-build-finished && \
#     echo "source /ros2_pkgs_ws/install/local_setup.sh" >> /root/.bashrc
   

# Install necessary system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gcc \
    libgtk2.0-dev \
    wget \
    curl \
    git \
    vim \
    libjpeg-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-pip \
    python3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libyaml-cpp-dev \
    libboost-all-dev \
    libhdf5-dev \
    libopenexr-dev \
    libx11-dev \
    libxi-dev \
    libgl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations einops kornia pyrender \
    meshcat webdataset omegaconf pypng seaborn wandb albumentations pycocotools ultralytics==8.0.120 \
    pillow urllib3 idna certifi python-dateutil pytz werkzeug click itsdangerous cycler kiwisolver kornia antlr4-python3-runtime \
    h5py kaolin joblib scipy plotly tenacity matplotlib fonttools contourpy --no-deps

# Install Eigen
RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && cd build && \
    cmake .. && make install && \
    rm -rf /eigen-3.4.0 eigen-3.4.0.tar.gz

# Install Pybind11
RUN cd / && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && make install && rm -rf /pybind11

# Install NVDiffRast
RUN cd / && git clone https://github.com/NVlabs/nvdiffrast && \
    cd nvdiffrast && python3 -m pip install . && rm -rf /nvdiffrast

# Install PyTorch3D
RUN git clone --recursive https://github.com/facebookresearch/pytorch3d.git && \
    cd pytorch3d && pip3 install ninja && python3 setup.py install && rm -rf /pytorch3d

# Install additional dependencies
RUN pip3 install torchvision==0.16.0 requests pandas dash flask --no-deps

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV SHELL=/bin/bash
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
RUN ln -sf /bin/bash /bin/sh

RUN pip install --force-reinstall "omegaconf>=2.2.3" "antlr4-python3-runtime==4.9.3"

RUN python3 -m pip install kornia kornia-rs scikit-learn --no-dependencies
RUN python3 -m pip install scikit-learn 
RUN python3 -m pip install transformers -U
