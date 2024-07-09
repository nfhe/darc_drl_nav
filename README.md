# P_EEMEF_DARC

## Project Description

P_EEMEF_DARC is an advanced robotics project focusing on autonomous navigation and collision avoidance using deep reinforcement learning. The project integrates various technologies, including ROS Melodic, TensorFlow, PyTorch, and Keras, to achieve efficient and reliable robot control. The main purpose is to provide a robust framework for TurtleBot3 to navigate complex environments autonomously.

**Demo Video:** [Watch Here](https://youtu.be/MnyCGTHqN8g)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following software installed on your machine:

- Ubuntu 18.04
- ROS Melodic
- TensorFlow-GPU (1.13.1 or 1.14.0)
- PyTorch (1.4)
- Keras (2.3.1)
- NumPy (1.16.6)

### Installation Steps

1. Activate your Python environment:
    ```bash
    conda activate <environment_name>
    source activate <environment_name>
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Install ROS Melodic and other dependencies:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_melodic.sh && chmod 755 ./install_ros_melodic.sh && bash ./install_ros_melodic.sh
    sudo apt-get install ros-melodic-joy ros-melodic-teleop-twist-joy ros-melodic-teleop-twist-keyboard ros-melodic-laser-proc ros-melodic-rgbd-launch ros-melodic-depthimage-to-laserscan ros-melodic-rosserial-arduino ros-melodic-rosserial-python ros-melodic-rosserial-server ros-melodic-rosserial-client ros-melodic-rosserial-msgs ros-melodic-amcl ros-melodic-map-server ros-melodic-move-base ros-melodic-urdf ros-melodic-xacro ros-melodic-compressed-image-transport ros-melodic-rqt-image-view ros-melodic-gmapping ros-melodic-navigation ros-melodic-interactive-markers
    ```

4. Clone the necessary repositories and compile the workspace:
    ```bash
    cd ~/catkin_nav/src/
    git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
    git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
    git clone https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance.git
    cd ~/catkin_nav && catkin_make
    echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
    source ~/.bashrc
    ```

## Usage

### 1. Launch Gazebo World

To start the simulation in Gazebo, use the following commands to launch different environments:

1.1. Launch Corridor 1 World
```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/3_corridor.world'
```

1.2. Launch Corridor 2 World
```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/2_corridor.world'
```

### 2. Collect Priority Excellent Experience Data

```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/record_maze.world'
```

### 3. Train and Test

3.1. Train and Play with DARC Using Human Data

1. Open the environment:
    ```bash
    cd ~/catkin_nav/src/turtlebot3_darc_nav/turtlebot_nav/scripts/record_data
    ```

2. Open two terminals and run the following scripts:
    ```bash
    python record_control.py
    python record_data.py
    ```

3.2. Training

1. Open the environment:
    ```bash
    cd ~/catkin_nav/src/turtlebot3_darc_nav/turtlebot_nav/scripts
    ```

2. Open the terminal and run one of the following Python files:
    ```bash
    python darc_turtlebot3_original_darc.py
    python p_darc_turtlebot3_original.py
    python turtlebot_turtlebot3_peemef_darc_env.py
    ```

## Authors

nfhe
