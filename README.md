# DAWGE
This is the learning and engineering based implementation of a manipulation framework built with A1 Quadruped robot. ![unitree_image](https://unitreerobotics.net/wp-content/uploads/2021/12/Unitree-A1-RobotDogs.jpg)

## Installation
### Installing the ROS package
After installing `unitree_ros` package by following the instructions [here](https://github.com/unitreerobotics/unitree_ros). You can install the ROS package of `DAWGE` by running the following command from the base of the repository:
```
catkin_make
source devel/setup.bash
```
With this the package `dawge_planner` will be installed.

### Installing the learning packages
You can create the necessary conda environment `dawge` by running the command:
```
conda env create -f dawge_env.yml
```

## Running
### Collecting data
In order to move the A1 robot with keyboard arrows, after starting the ROS master with `roscore`, one should run:
```
rosrun dawge_planner teleop
```
Then 
```
rosrun dawge_planner save_all
```
will save state based and image based information of the framework.

