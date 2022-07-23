# Script to test the command files that are saved

from turtle import forward
import cv2
import glob
import matplotlib
import math
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle

from cv2 import aruco

from unitree_legged_msgs.msg import HighCmd, HighState

from matplotlib.animation import FuncAnimation, FFMpegWriter

from contrastive_learning.tests.animate_markers import AnimateMarkers
from contrastive_learning.tests.animate_rvec_tvec import AnimateRvecTvec

CAMERA_INTRINSICS = np.array([[612.82019043,   0.        , 322.14050293],
                              [  0.        , 611.48303223, 247.9083252 ],
                              [  0.        ,   0.        ,   1.        ]])

# Function to draw box and dog position and applied action
def plot_corners_state(ax, curr_pos, plot_action, actions=None, fps=15, color_scheme=1): # Color scheme is to have an alternative color for polygon colors
    min_x = -1.0 # Minimums and maximums around all the data
    max_x = 1228.0
    min_y = -1.0
    max_y = 716.0
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(min_x, max_x)

    for j in range(2):
        # Get the current position for box and the dog
        curr_polygon = curr_pos[j*4:(j+1)*4,:]
        mean_x, mean_y = curr_polygon[:,0].mean(), curr_polygon[:,1].mean()
        right_top_x, right_top_y = curr_polygon[0,0], curr_polygon[0,1]
        right_bot_x, right_bot_y = curr_polygon[1,0], curr_polygon[1,1]
        
        if j == 0:
            # Show the box position
            if color_scheme == 1:
                box_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='b', fill=False, label='Box Position')
            else:
                box_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='m', fill=False, label='Predicted Box Position')
            ax.add_patch(box_poly)
        else:
            # Show the dog position
            if color_scheme == 1:
                dog_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='g', fill=False, label='Dog Position')
            else:
                dog_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='k', fill=False, label='Predicted Dog Position')
            ax.add_patch(dog_poly)
            
            if plot_action: # Action can be none for the other positions
                # Set the action direction
                front_x, front_y = ( right_top_x + right_bot_x ) / 2, ( right_top_y + right_bot_y ) / 2
                curr_dir = np.arctan2(front_y-mean_y, front_x-mean_x )
                pred_dir = np.arctan2(front_y-mean_y, front_x-mean_x)
                action = actions[0]
                pred_action = actions[1]

                # Show the current action 
                forward_speed = action[0]
                rotate_speed = action[1]
                curr_dir -= rotate_speed
                action_x = forward_speed * math.sin(curr_dir) * 400 # 250 is only for scaling
                action_y = forward_speed * math.cos(curr_dir) * 400
                action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='c', label='Actual Action') # - is for drawing purposes
                ax.add_patch(action_arr)
            
                # Show the predicted action
                forward_speed = pred_action[0]
                rotate_speed = pred_action[1]
                pred_dir -= rotate_speed
                action_x = forward_speed * math.sin(pred_dir) * 400 # 250 is only for scaling
                action_y = forward_speed * math.cos(pred_dir) * 400
                action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='m', label='Predicted Action') # - is for drawing purposes
                ax.add_patch(action_arr)

            ax.plot()
            ax.legend()

# Draw the boxes with the same way as rvec tvec plotting
def plot_corners(ax, curr_pos, use_img=False, img=None, use_frame_axis=False, frame_axis=None, plot_action=False, actions=None, color_scheme=1):
    # actions: [action, pred_action]
    if plot_action:
        action = actions[0]
        pred_action = actions[1]
        dir = 0
        pred_dir = 0
        action_pos = (1100,600)

    img_shape = (720, 1280, 3)
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False: # use img is when two plots are drawn on top of each other
        img = ax.imshow(blank_image.copy())

    # Plot the boxed
    for j in range(2):
        curr_polygon = curr_pos[j*4:(j+1)*4,:]
#         print('curr_polygon.shape: {}'.format(curr_polygon.shape))
        if j == 0: # Show the box position
            if color_scheme == 1:
                box_color = (51,102,0)
            else:
                box_color = (102,204,0)
            if use_frame_axis:
                frame_axis = cv2.polylines(frame_axis.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                       isClosed=True, color=box_color, thickness=3)
            else:
                frame_axis = cv2.polylines(blank_image.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                        isClosed=True, color=box_color, thickness=3)

        else:
            if color_scheme == 1:
                dog_color = (0,0,153)
            else:
                dog_color = (0,0,255)

            frame_axis = cv2.polylines(frame_axis.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                       isClosed=True, color=dog_color, thickness=3)

    if plot_action:
        # Actual action
        forward_speed = action[0]
        rotate_speed = action[1]
        dir -= rotate_speed
        action_x = forward_speed * math.sin(dir) * 500 # 250 is only for scaling
        action_y = forward_speed * math.cos(dir) * 500
        frame_axis = cv2.arrowedLine(frame_axis.copy(), action_pos,
                                        (int(action_pos[0]+action_x), int(action_pos[1]-action_y)), # Y should be removed from the action
                                        color=(0,200,200), thickness=3)

        
        # Draw an ellipse to show the rotate_speed more thoroughly
        ellipse_pos = (900,600)
        axesLength = (50, 50)
        angle = 0
        startAngle = 0
        endAngle = rotate_speed * (1080. / np.pi)
        frame_axis = cv2.ellipse(frame_axis.copy(), ellipse_pos, axesLength,
                angle, startAngle, endAngle, color=(0,200,200), thickness=3)


        # Predicted action
        ellipse_pos = (800,600)
        forward_speed = pred_action[0]
        rotate_speed = pred_action[1]
        endAngle = rotate_speed * (1080. / np.pi)
        pred_dir -= rotate_speed
        action_x = forward_speed * math.sin(pred_dir) * 500 # 250 is only for scaling
        action_y = forward_speed * math.cos(pred_dir) * 500
        frame_axis = cv2.arrowedLine(frame_axis.copy(), action_pos,
                                        (int(action_pos[0]+action_x), int(action_pos[1]-action_y)), # Y should be removed from the action
                                        color=(104,43,159), thickness=3)
        frame_axis = cv2.ellipse(frame_axis.copy(), ellipse_pos, axesLength,
                angle, startAngle, endAngle, color=(104,43,159), thickness=3)

    img.set_array(frame_axis) # If use_img is true then img will not be none
    ax.plot()

    return img, frame_axis

# Function to draw box and dog position and applied action
def plot_rvec_tvec(ax, curr_pos, use_img=False, img=None, plot_action=False, actions=None): # Color scheme is to have an alternative color for polygon colors
    # actions: [action, pred_action]
    if plot_action:
        action = actions[0]
        pred_action = actions[1]
        dir = 0
        pred_dir = 0
        action_pos = (1100,600)

    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False:
        img_shape = (720, 1280, 3)
        img = ax.imshow(blank_image.copy())

    for j in range(2):
        curr_rvec_tvec = curr_pos[j*6:(j+1)*6]
        if j == 0:
            frame_axis = aruco.drawAxis(blank_image.copy(),
                CAMERA_INTRINSICS,
                np.zeros((5)),
                curr_rvec_tvec[:3], curr_rvec_tvec[3:],
                0.01)
        else:
            frame_axis = aruco.drawAxis(frame_axis.copy(),
                CAMERA_INTRINSICS,
                np.zeros((5)),
                curr_rvec_tvec[:3], curr_rvec_tvec[3:],
                0.01)

    if plot_action:
        # Actual action
        forward_speed = action[0]
        rotate_speed = action[1]
        dir -= rotate_speed
        action_x = forward_speed * math.sin(dir) * 500 # 250 is only for scaling
        action_y = forward_speed * math.cos(dir) * 500
        # action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='c', label='Actual Action') # - is for drawing purposes
        frame_axis = cv2.arrowedLine(frame_axis.copy(), action_pos,
                                        (int(action_pos[0]+action_x), int(action_pos[1]-action_y)), # Y should be removed from the action
                                        color=(0,200,200), thickness=3)

        # Predicted action
        forward_speed = pred_action[0]
        rotate_speed = pred_action[1]
        pred_dir -= rotate_speed
        action_x = forward_speed * math.sin(pred_dir) * 500 # 250 is only for scaling
        action_y = forward_speed * math.cos(pred_dir) * 500
        frame_axis = cv2.arrowedLine(frame_axis.copy(), action_pos,
                                        (int(action_pos[0]+action_x), int(action_pos[1]-action_y)), # Y should be removed from the action
                                        color=(104,43,159), thickness=3)

    img.set_array(frame_axis) # If use_img is true then img will not be none
    ax.plot()

    return img

    
if __name__ == "__main__":
    demo_name = 'box_marker_7'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests/animations'
    # dump_file = '{}_corners.mp4'.format(demo_name)

    data_dirs = glob.glob("/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_marker_*")
    dump_file_mult = 'all_markers_test.mp4'
    fps = 15

    AnimateMarkers(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = '{}_corners.mp4'.format(demo_name),
        fps = fps,
        mult_traj = False,
        show_predicted_action=False
    )

    AnimateRvecTvec(
        data_dir = data_dir, 
        dump_dir = dump_dir,
        dump_file = '{}_rvec_tvec.mp4'.format(demo_name),
        fps = fps
    )

    # AnimateMarkers(
    #     data_dir = data_dir, 
    #     dump_dir = dump_dir, 
    #     dump_file = dump_file, 
    #     # dump_file = dump_file,
    #     fps = fps,
    #     mult_traj = False,
    #     show_predicted_action = True 
    # )

    # AnimatePosFrame(
    #     data_dir = data_dir, 
    #     dump_dir = dump_dir, 
    #     dump_file = f'pos_{dump_file}', 
    #     fps = fps
    # )
