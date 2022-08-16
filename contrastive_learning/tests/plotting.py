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
import scipy.stats as stats

from cv2 import aruco

from unitree_legged_msgs.msg import HighCmd, HighState

from matplotlib.animation import FuncAnimation, FFMpegWriter

CAMERA_INTRINSICS = np.array([[612.82019043,   0.        , 322.14050293],
                              [  0.        , 611.48303223, 247.9083252 ],
                              [  0.        ,   0.        ,   1.        ]])


def plot_corners(ax, curr_pos, use_img=False, img=None, use_frame_axis=False, frame_axis=None, plot_action=False, actions=None, color_scheme=1):
    # actions: [action, pred_action]
    # curr_pos.shape: (8,2)
    
    img_shape = (720, 1280, 3)
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False: # use img is when two plots are drawn on top of each other
        img = ax.imshow(blank_image.copy())

    # Plot the boxed
    for j in range(2):
        curr_polygon = curr_pos[j*4:(j+1)*4,:] # First position is box's the second one is dog's
        if color_scheme == 1: # Actual corners - blue ones
            box_color = (0,0,255)
            dog_color = (0,0,153)
        else: # Predicted corners - predicted green ones
            box_color = (102,204,0)
            dog_color = (51,102,0)

        if j == 0: # Show the box position
            if use_frame_axis:
                frame_axis = cv2.polylines(frame_axis.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                       isClosed=False, color=box_color, thickness=3)
            else:
                frame_axis = cv2.polylines(blank_image.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                        isClosed=False, color=box_color, thickness=3)
        else:
            frame_axis = cv2.polylines(frame_axis.copy(), np.int32([curr_polygon.reshape((-1,1,2))]),
                                       isClosed=False, color=dog_color, thickness=3)

    if plot_action:
        frame_axis = plot_actions(actions, frame_axis)

    # print('img: {}'.format(img))
    img.set_array(frame_axis) # If use_img is true then img will not be none
    ax.plot()

    return img, frame_axis

# TODO: Make these two functions the same way
# Function to draw box and dog position and applied action
def plot_rvec_tvec(ax, curr_pos, use_img=False, img=None, plot_action=False, actions=None): # Color scheme is to have an alternative color for polygon colors
    # actions: [action, pred_action]

    img_shape = (720, 1280, 3)
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False:
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
        frame_axis = plot_actions(actions, frame_axis)
        
    img.set_array(frame_axis) # If use_img is true then img will not be none
    ax.plot()

    return img, frame_axis


def plot_mean_rot(ax, curr_pos, use_img=False, img=None, use_frame_axis=False, frame_axis=None, plot_action=False, actions=None, color_scheme=1):
    # actions: [action, pred_action]
    # curr_pos.shape: (6) [box_mean_x, box_mean_y, box_rot, dog_mean_x, dog_mean_y, dog_rot]

    img_shape = (720, 1280, 3)
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False: # use img is when two plots are drawn on top of each other
        img = ax.imshow(blank_image.copy())

    if color_scheme == 1: # Actual corners - blue ones
        box_color = (0,0,255)
        dog_color = (0,0,153)
    else: # Predicted corners - predicted green ones
        box_color = (102,204,0)
        dog_color = (51,102,0)

    # Plot the rotation and the mean
    for j in range(2):
        curr_obs_pos = curr_pos[j*3:(j+1)*3]
        curr_mean = (int(curr_obs_pos[0]), int(curr_obs_pos[1]))
        curr_rot = curr_obs_pos[2]

        line_len = 50
        curr_end_point = (int(curr_mean[0] + np.cos(curr_rot) * line_len),
                          int(curr_mean[1] - np.sin(curr_rot) * line_len)) # (y starts from top left corner)

        if j == 0: # Show the box position
            if use_frame_axis:
                frame_axis = cv2.line(frame_axis.copy(), curr_mean, curr_end_point,
                                      color=box_color, thickness=3)
            else:
                frame_axis = cv2.line(blank_image.copy(), curr_mean, curr_end_point,
                                      color=box_color, thickness=3)

        else: # We will already have a frame axis given - show the dog position
            frame_axis = cv2.line(frame_axis.copy(), curr_mean, curr_end_point,
                                  color=dog_color, thickness=3)

    if plot_action:
        frame_axis = plot_actions(actions, frame_axis)

    img.set_array(frame_axis)
    ax.plot()

    return img, frame_axis

def plot_actions(actions, frame_axis):
    action_pos = (1100,600)

    # Actual action
    action = actions[0]
    forward_speed = action[0]
    rotate_speed = -action[1]
    dir = rotate_speed
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

    show_predicted_action = len(actions) > 1
    if show_predicted_action:
        # Predicted action
        ellipse_pos = (800,600)
        pred_action = actions[1]
        forward_speed = pred_action[0]
        rotate_speed = -pred_action[1]
        endAngle = rotate_speed * (1080. / np.pi)
        pred_dir = rotate_speed
        action_x = forward_speed * math.sin(pred_dir) * 500 # 250 is only for scaling
        action_y = forward_speed * math.cos(pred_dir) * 500
        frame_axis = cv2.arrowedLine(frame_axis.copy(), action_pos,
                                        (int(action_pos[0]+action_x), int(action_pos[1]-action_y)), # Y should be removed from the action
                                        color=(104,43,159), thickness=3)
        frame_axis = cv2.ellipse(frame_axis.copy(), ellipse_pos, axesLength,
                angle, startAngle, endAngle, color=(104,43,159), thickness=3)

    return frame_axis

# Diffusion related plots
def plot_gaus_dist(x, ax, label):
    traj_len = len(x)

    mean = x.mean()
    std = x.std()

    lin_range = np.linspace(mean-std*2, mean+std*2, traj_len)
    ax.plot(lin_range, stats.norm.pdf(lin_range, mean, std), label=label)
    ax.legend()


def plot_data(x, ax, label):
    traj_len = len(x)
    ax.plot(x, np.random.rand(traj_len), 'o', label=label)
    ax.legend()