#!/usr/bin/env python3

# ROS related imports
import rospy
from unitree_legged_msgs.msg import HighState, HighCmd # TODO: Check if this works!! 

# Script that implements the base class for any high level task to be used with DAWGE

class HighLevelTask():
    # High commands will be published and high states will be listened
    def __init__(self, high_cmd_topic, high_state_topic, rate):
        # Initialize ros node
        rospy.init_node("high_level_task")

        # Initialize the subscriber
        rospy.Subscriber(high_state_topic, HighState, self.high_state_cb)
        self.high_state_msg = None

        # Initialize the publisher
        self.high_cmd_pub = rospy.Publisher(high_cmd_topic, HighCmd, queue_size=10)
        self.high_cmd_msg = HighCmd()

        # Initialize the rate for running loop
        self.rate_count = rate
        self.rate = rospy.Rate(rate) # NOTE: Check if this is the way to go
        self.motion_time = 0

    def high_state_cb(self, data):
        self.high_state_msg = data
        # print('self.high_state_msg: {}'.format(data)) # TODO: delete this once you make sure everything works well

    def run(self):
        while not rospy.is_shutdown():
            self.motion_time += 1

            if self.motion_time > self.rate_count*4 and self.is_initialized():
                self.update_high_cmd()

            # print('Publishing: rotateSpeed: {}, forwardSpeed: {}'.format(
            #     self.high_cmd_msg.rotateSpeed, self.high_cmd_msg.forwardSpeed
            # ))
            self.high_cmd_pub.publish(self.high_cmd_msg)

            self.rate.sleep()


    def update_high_cmd(self):
        pass

    def is_initialized(self):
        pass