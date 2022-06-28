#!/usr/bin/env python3

# Script to save the high level commands that are sent to the robot
# Even though the commands will come frequently they will be added to actions array with
# same frequency as the save_stream
# TODO: get cam_fps from launch file

import pickle
import rospy
import signal
import os

from datetime import datetime

from unitree_legged_msgs.msg import HighCmd, HighState

class SaveRobot(): # Any data related to robot - high commands and high states should be saved
    # High commands will be published and high states will be listened
    def __init__(self, data_dir, high_cmd_topic, high_state_topic, fps=30):
        # Initialize ros node
        rospy.init_node("dawge_robot_save", disable_signals=True)

        # Initialize the subscriber
        rospy.Subscriber(high_state_topic, HighState, self.high_state_cb)
        rospy.Subscriber(high_cmd_topic, HighCmd, self.high_cmd_cb)
        self.high_cmd_msg = None
        self.high_state_msg = None

        # Initialize the arrays
        self.data_dir = data_dir
        self.high_cmds = []
        self.high_states = [] # For now we won't save the time stamps

        # Initialize the rate for running loop
        self.rate = rospy.Rate(fps) # NOTE: Check if this is the way to go

        signal.signal(signal.SIGINT, self.end_signal_handler)

    def high_state_cb(self, data):
        self.high_state_msg = data

    def high_cmd_cb(self, data):
        self.high_cmd_msg = data 

    def run(self):
        while not rospy.is_shutdown():
            self.high_cmds.append(self.high_cmd_msg)
            self.high_states.append(self.high_state_msg)

            self.rate.sleep() # This will help synchronize messages

    # Dump the commands and states as a pickle file
    def dump(self):
        with open('{}/commands.pickle'.format(self.data_dir), 'wb') as pkl:
            pickle.dump(self.high_cmds, pkl, pickle.HIGHEST_PROTOCOL)
        with open('{}/states.pickle'.format(self.data_dir), 'wb') as pkl:
            pickle.dump(self.high_states, pkl, pickle.HIGHEST_PROTOCOL)


    def end_signal_handler(self, signum, frame):
        self.dump()
        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)

if __name__ == '__main__':
    
    now = datetime.now()
    time_str = now.strftime('%d%m%Y_%H%M%S')
    data_dir = '{}/{}'.format(
        '/home/irmak/Workspace/DAWGE/src/dawge_planner/data',
        time_str
    )

    data_saver = SaveRobot(
        video_dir=data_dir,
        high_cmd_topic='dawge_high_cmd',
        high_state_topic='dawge_high_state'
    ) 

    data_saver.run()

