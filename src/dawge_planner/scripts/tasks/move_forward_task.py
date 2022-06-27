#!/usr/bin/env python3

# ROS related imports
import rospy 
from unitree_legged_msgs.msg import HighCmd, HighState

# Custom imports 
from dawge_planner.scripts.tasks.high_lvl_task import HighLevelTask

# During this task the robot should always move forward
# so that we can check if forward position has passed the desired dist
class MoveForwardTask(HighLevelTask):
    def __init__(self, high_cmd_topic, high_state_topic, rate, des_dist, max_speed):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        self.max_speed = max_speed # The velocity will be lower and lower as we get closer to the desired distance
        self.desired_dist = des_dist # Amount of meters for the robot to move forward
        self.reached = False

    # We only need to implement updating high level task part
    def update_high_cmd(self):
        print('reached: {} - forwardPosition: {} - sidePosition: {}'.format(
            self.reached, self.high_state_msg.forwardPosition, self.high_state_msg.sidePosition
        ))

        if not self.reached and self.high_state_msg.forwardPosition < self.desired_dist - 1e-2: # Check if I should add data here - look at the logs of task_base
            self.high_cmd_msg.mode = 2
            # left_dist = self.desired_dist - self.high_state_msg.forwardPosition
            # self.high_cmd_msg.forwardSpeed = self.max_speed * (left_dist / self.desired_dist) # The velocity will gradually get smaller and smaller
            self.high_cmd_msg.forwardSpeed = self.max_speed

            # Control the side position of the robot as well - basic linear control
            # if self.high_state_msg.sidePosition > 0:
            self.high_cmd_msg.sideSpeed = -self.high_state_msg.sidePosition
            
        else:
            self.high_cmd_msg.forwardSpeed = 0
            self.high_cmd_msg.mode = 1
            self.reached = True

    def is_initialized(self):
        return not self.high_state_msg is None

if __name__ == "__main__":
    task = MoveForwardTask(
        high_cmd_topic="dawge_high_cmd",
        high_state_topic="dawge_high_state",
        rate=100,
        des_dist=1.0,
        max_speed=0.1
    )

    task.run()
