# DAWGE
This is the learning and engineering based implementation of a manipulation framework built with [A1 Quadruped robot](https://m.unitree.com/a1/).
This repository includes teleoperation of a quadruped dog and different training and deployment frameworks to make this dog push objects to desired positions.
Experimented frameworks for prediction of the next observations are as follows:
1. **Image based prediciton framework**: Here an encoder trained with self-supervised learning algorithms where contrastive loss is used for training. Here positive pairs are the predicted representation of the next observation and actual next observation.
2. **State based prediction framework**: Here, aruco markers are places on both the dog and the manipulation object and the next states of each object is predicted. We used classical supervised learning algorithms and diffusion models to predict the next state.
3. **Inverse models**: To predict the action applied between the current and next state we applied supervised inverse models. 

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

### Training

All the configurations of the training can be modified [here](https://github.com/irmakguzey/DAWGE/blob/main/contrastive_learning/configs/train.yaml).
Agents to use while training are the frameworks tried and implemented [here](https://github.com/irmakguzey/DAWGE/tree/main/contrastive_learning/models/agents). Agents refer to the main experimented frameworks mentioned above. After having the desired modifications one can train the frameworks by simply running: 
```
conda activate dawge
python train.py
```

### Deployment on the robot
Different scripts to run models on the robot can be found on the ROS package, [here](https://github.com/irmakguzey/DAWGE/tree/main/src/dawge_planner/scripts/dawge_planner/deploy_models). 
One can run these by running: 
```
conda activate dawge
python <desired_model_deployment>.py
```

