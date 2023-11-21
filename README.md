# Project Name：P_EEMEF_DARC

## Project description
Here is the description of the project. Please provide a detailed explanation of what your project is about, its functionality, and purpose.
Demo Video: (https://youtu.be/MnyCGTHqN8g)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

```bash
Ubuntu 18.04
ROS Melodic
tensorflow-gpu == 1.13.1 or 1.14.0
torch==1.4
Keras == 2.3.1
numpy == 1.16.6
1.conda activate name python==2.7
2.source activate name
3.pip install -r requirements.txt
```

# Install
Describe how to install your project. List any required dependencies and installation steps.
1.The next step is to install dependent packages for TurtleBot3 control on Remote PC. For more details, please refer to (https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/).
```bash
2. sudo apt-get update
3.sudo apt-get upgrade
4.wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_melodic.sh && chmod 755 ./install_ros_melodic.sh && bash ./install_ros_melodic.sh

5.sudo apt-get install ros-melodic-joy ros-melodic-teleop-twist-joy ros-melodic-teleop-twist-keyboard ros-melodic-laser-proc ros-melodic-rgbd-launch ros-melodic-depthimage-to-laserscan ros-melodic-rosserial-arduino ros-melodic-rosserial-python ros-melodic-rosserial-server ros-melodic-rosserial-client ros-melodic-rosserial-msgs ros-melodic-amcl ros-melodic-map-server ros-melodic-move-base ros-melodic-urdf ros-melodic-xacro ros-melodic-compressed-image-transport ros-melodic-rqt-image-view ros-melodic-gmapping ros-melodic-navigation ros-melodic-interactive-markers
6.cd ~/catkin_nav/src/
7.git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
8.git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
9.git clone https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance.git
10.cd ~/catkin_nav && catkin_make
11.echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
```

## Start
#1.Start gazebo world

### To launch the corridor 1 world
```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/3_corridor.world'
```

### To launch the corridor 2 world
```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/2_corridor.world'
```

### 2.Collect priority excellent experience data
```bash
roslaunch turtlebot_darc turtlebot3_empty_world.launch world_file:='/home/he/catkin_nav/src/turtlebot3_darc_nav/turtlebot_darc/world/record_maze.world'
```
### 3.For train and test
#### For train and play with darc with human data.
(1).open the environment.
```bash
 cd ~/catkin_nav/src/turtlebot3_darc_nav/turtlebot_nav/scripts/record_data
 ```
(2).Open two terminals and run the py
```bash
python record_control.py
```
```bash
python record_data.py
```

For training
 (1).open the environment.
```bash
cd ~/catkin_nav/src/turtlebot3_darc_nav/turtlebot_nav/scripts
```
(2).Open the terminal and run one of the following Python files

```bash
python darc_turtlebot3_original_darc.py
```

```bash
python p_darc_turtlebot3_original.py
```

```bash
python turtlebot_turtlebot3_peemef_darc_env.py
```



## Authors
nfhe


