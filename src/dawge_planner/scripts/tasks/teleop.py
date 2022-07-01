#!/usr/bin/env python3

# Script to listen to teleop cmd_vel topic and sends high cmds accordingly

# ROS related imports
import rospy 

from geometry_msgs.msg import Twist
from unitree_legged_msgs.msg import HighCmd, HighState

# Custom imports 
# from high_lvl_task import HighLevelTask
from high_lvl_task import HighLevelTask


class TeleopTask(HighLevelTask):
    def __init__(self, high_cmd_topic, high_state_topic, rate, twist_topic, lin_vel, ang_vel):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        self.lin_vel = lin_vel
        self.ang_vel = ang_vel 

        # Initialize the listener for the cmd_vel topic
        self.twist_msg = None
        rospy.Subscriber(twist_topic, Twist, self.twist_cb)

    def twist_cb(self, data): 
        self.twist_msg = data 

    def update_high_cmd(self):
        if self.twist_msg is None:
            return

        self.high_cmd_msg.mode = 2

        # Set the angular velocity - we will use set linear and angular velocity
        self.high_cmd_msg.rotateSpeed = 0
        if self.twist_msg.angular.z > 0:
            self.high_cmd_msg.rotateSpeed = self.ang_vel 
        elif self.twist_msg.angular.z < 0:
            self.high_cmd_msg.rotateSpeed = -self.ang_vel             

        # Set the linear velocity 
        self.high_cmd_msg.forwardSpeed = 0
        if self.twist_msg.linear.x > 0:
            self.high_cmd_msg.forwardSpeed = self.lin_vel 
        elif self.twist_msg.linear.x < 0:
            self.high_cmd_msg.forwardSpeed = -self.lin_vel 

    def is_initialized(self):
        return not self.twist_msg is None


if __name__ == "__main__":
    task = TeleopTask(
        high_cmd_topic="dawge_high_cmd",
        high_state_topic="dawge_high_state",
        rate=100,
        twist_topic="cmd_vel",
        lin_vel=0.15,
        ang_vel=0.3
    )

    task.run()