#!/usr/bin/env python

# ROS related imports
import rospy 
from unitree_legged_msgs.msg import HighCmd, HighState

# Custom imports 
from task_base import HighLevelTask

# During this task the robot should always move forward
# so that we can check if forward position has passed the desired dist
class MoveForwardTask(HighLevelTask):
    def __init__(self, high_cmd_topic, high_state_topic, rate, des_dist):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        self.max_speed = 0.3 # The velocity will be lower and lower as we get closer to the desired distance
        self.desired_dist = des_dist # Amount of meters for the robot to move forward
        self.reached = False

    # We only need to implement updating high level task part
    def update_high_cmd(self):
        print('reached: {} - self.high_state_msg.forwardPosition: {}'.format(
            self.reached, self.high_state_msg.forwardPosition
        ))

        if not self.reached and self.high_state_msg.forwardPosition < self.desired_dist - 1e-2: # Check if I should add data here - look at the logs of task_base
            self.high_cmd_msg.mode = 2
            # left_dist = self.desired_dist - self.high_state_msg.forwardPosition
            # self.high_cmd_msg.forwardSpeed = self.max_speed * (left_dist / self.desired_dist) # The velocity will gradually get smaller and smaller
            self.high_cmd_msg.forwardSpeed = 0.3
        else:
            self.high_cmd_msg.forwardSpeed = 0
            self.high_cmd_msg.mode = 1
            self.reached = True

if __name__ == "__main__":
    task = MoveForwardTask(
        high_cmd_topic="dawge_high_cmd",
        high_state_topic="dawge_high_state",
        rate=100,
        des_dist=1.0
    )

    task.run()
