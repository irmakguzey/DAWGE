#!/usr/bin/env python3

# Script to move dawge robot by using moveit

import rospy 
import geometry_msgs.msg 

from std_msgs.msg import String 
from moveit_commander.conversions import pose_to_list 

import sys
import copy 
import moveit_commander 
import moveit_msgs.msg 

from math import pi, tau, dist, fabs, cos 

FEET_LINKS = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Define the indices for robot legs 
FL_ = 0
FR_ = 1
RL_ = 2 
RR_ = 3

# Joint indices
FL_0 = 0
FL_1 = 1
FL_2 = 2 

FR_0 = 3 
FR_1 = 4
FR_2 = 5 

RL_0 = 6
RL_1 = 7
RL_2 = 8

RR_0 = 9
RR_1 = 10
RR_2 = 11

def all_close(goal, actual, tolerance): 
    '''
    Method to check if goal and actual poses are closer than a given tolerance
    '''

    if type(goal) is list: 
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance: 
                return False 

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)

        # Euclidean distance 
        d = dist((x1, y1, z1), (x0, y0, z0))
        cos_phi_half = fabs(qx0*qx1 + qy0*qy1 + qz0*qz1 + qw0*qw1) 

        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True 

class DawgeMoveItWrapper:
    # Wrapper to control dawge with moveit interface
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("dawge_moveit_wrapper", anonymous=True)

        # Instantiate a RobotCommander object. Provides information such as the 
        # robot's kinematic model and the robot's current joint states
        self.robot = moveit_commander.RobotCommander() 

        # Start a PlanningSceneInterface object. This provides a remote interface for
        # for getting, setting and updating the robot's internal understanding of the 
        # surrounding world.
        self.scene = moveit_commander.PlanningSceneInterface() 


        # Initialize a MoveGroupCommander object. This obhect is an interface to a 
        # planning group (group of joints). In this tutorial the group is the legs of 
        # the Dawge robot, so we initialize multiple planning groups and set the name to
        # fr/fl/rr/rl_leg. 
        self.group_names = ["fl_leg", "fr_leg", "rl_leg", "rr_leg"]
        self.move_groups = [] 
        for i in range(len(self.group_names)):
            self.move_groups.append(moveit_commander.MoveGroupCommander(self.group_names[i]))

        # Create a DisplayTrajectory ROS publisher which is used to display 
        # trajectories in Rviz
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # Some planning names 
        fl_planning_frame = self.move_groups[FL_].get_planning_frame() 
        print('============== FL Planning Frame: {}'.format(fl_planning_frame)) 

        # End effector link in FL leg 
        fl_eef_link = self.move_groups[FL_].get_end_effector_link()
        print('============== FL End Effector link: {}'.format(fl_eef_link))

        # Get all group names in the robot 
        all_group_names = self.robot.get_group_names()
        print('============== ALL planning groups: {}'.format(all_group_names))

        print("============== Printing the robot state")
        print(self.robot.get_current_state())
        print('')

        # Get the position of the FL foot
        all_link_names = self.robot.get_link_names()
        print('============== All link names: {}'.format(all_link_names))

        # Get the poses of the feet
        print("============== POSES OF ALL FEET:")
        for i in range(len(FEET_LINKS)):
            foot_link = self.robot.get_link(FEET_LINKS[i])
            print('Feet {} Pose: {}'.format(FEET_LINKS[i], foot_link.pose()))
        print() 



    def go_to_pose_goal(self):
        # TODO: add pose and leg index as the parameter
        # For now this is done only for the front left leg
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        fl_move_group = self.move_groups[FL_]
        fl_link = self.robot.get_link(FEET_LINKS[FL_])

        curr_pose = fl_link.pose().pose


        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = copy.deepcopy(curr_pose)
        # pose_goal.orientation.w = 1.0
        pose_goal.position.x += 0.1
        pose_goal.position.y += 0.1

        # fl_move_group.set_pose_target(pose_goal)
        fl_move_group.set_joint_value_target(curr_pose, "FL_foot", True)

        plan = fl_move_group.plan()
        print('plan: {}'.format(plan))

        self.execute_plan(FL_, plan[1])

        ## Now, we call the planner to compute the plan and execute it.
        # plan = fl_move_group.go(wait=True)
        # print('plan: {}'.format(plan))
        # Calling `stop()` ensures that there is no residual movement
        fl_move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        fl_move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_groups[FL_].get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def execute_plan(self, leg_index, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_groups[leg_index]

        ## BEGIN_SUB_TUTORIAL execute_plan
        ##
        ## Executing a Plan
        ## ^^^^^^^^^^^^^^^^
        ## Use execute if you would like the robot to follow
        ## the plan that has already been computed:
        move_group.execute(plan, wait=True)

        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail
        ## END_SUB_TUTORIAL
        
def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("Welcome to the MoveIt MoveGroup Python Interface Tutorial")
        print("----------------------------------------------------------")
        print("Press Ctrl-D to exit at any time")
        print("")
        input(
            "============ Press `Enter` to begin the tutorial by setting up the moveit_commander ..."
        )
        dawge_wrapper = DawgeMoveItWrapper()

        input(
            "============ Press `Enter` to execute a movement using a joint state goal ..."
        )
        # tutorial.go_to_joint_state()

        input("============ Press `Enter` to execute a movement using a pose goal ...")
        dawge_wrapper.go_to_pose_goal()

        # input("============ Press `Enter` to plan and display a Cartesian path ...")
        # cartesian_plan, fraction = tutorial.plan_cartesian_path()

        # input(
        #     "============ Press `Enter` to display a saved trajectory (this will replay the Cartesian path)  ..."
        # )
        # tutorial.display_trajectory(cartesian_plan)

        # input("============ Press `Enter` to execute a saved path ...")
        # tutorial.execute_plan(cartesian_plan)

        # input("============ Press `Enter` to add a box to the planning scene ...")
        # tutorial.add_box()

        # input("============ Press `Enter` to attach a Box to the Panda robot ...")
        # tutorial.attach_box()

        # input(
        #     "============ Press `Enter` to plan and execute a path with an attached collision object ..."
        # )
        # cartesian_plan, fraction = tutorial.plan_cartesian_path(scale=-1)
        # tutorial.execute_plan(cartesian_plan)

        # input("============ Press `Enter` to detach the box from the Panda robot ...")
        # tutorial.detach_box()

        # input(
        #     "============ Press `Enter` to remove the box from the planning scene ..."
        # )
        # tutorial.remove_box()

        print("============ Python tutorial demo complete!")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()




