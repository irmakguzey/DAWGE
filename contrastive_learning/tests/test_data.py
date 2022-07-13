# Script to test the command files that are saved

import cv2
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle

from cv2 import aruco

from unitree_legged_msgs.msg import HighCmd, HighState

from matplotlib.animation import FuncAnimation, FFMpegWriter

# TODO: add images as well
class AnimatePosFrame:
    def __init__(self, data_dir, dump_dir, dump_file, fps=15):
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=1, ncols=1)

        # Create the dump_dir if it doesn't exist
        os.makedirs(dump_dir, exist_ok=True)

        # Load the data
        with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
            self.commands = pickle.load(f)
        with open(os.path.join(data_dir, 'states.pickle'), 'rb') as f:
            self.states = pickle.load(f)
        assert self.commands is not None , "Commands are not loaded well"
        assert self.states is not None, "States are not loaded well"
        assert len(self.commands) == len(self.states), "Number of states and commands should be same"

        num_frames = len(self.commands)
        print('num_frames: {}'.format(num_frames))

        # Set the axes
        self.line, = self.axs.plot([], [])
        self.dir = 0
        self.fps = fps
        self.pos = np.zeros((num_frames, 2)) # This will be filled with forward and side position
        self.axs.set_ylim(-0.5, 3)
        self.axs.set_xlim(-0.5, 2)
        self.axs.set_title("Given Commands")

        # Create the animation object and save it
        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = num_frames
        )
        self.anim.save(os.path.join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('Animation saved to: {}'.format(dump_file))

    def init_fun(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):

        rotate_speed = self.commands[i].rotateSpeed / (self.fps/3) # Since the velocity is given in m/s (or rad/s)
        forward_speed = self.commands[i].forwardSpeed / (self.fps/3) # NOTE: /3 should be deleted normally

        self.dir -= rotate_speed

        action_x = forward_speed * math.sin(self.dir)
        action_y = forward_speed * math.cos(self.dir)
        
        if i > 0:
            self.pos[i][0] = self.pos[i-1][0] + action_x
            self.pos[i][1] = self.pos[i-1][1] + action_y
        else:
            self.pos[i][0] = action_x
            self.pos[i][1] = action_y
        print(f'Frame: {i} - pos: (x={self.pos[i][0]}, y={self.pos[i][1]})')
        
        self.line.set_data(self.pos[:i,0], self.pos[:i,1])

        return self.line,

class AnimateMarkers:
    def __init__(self, data_dir, dump_dir, dump_file, fps):
        # Create the figure to draw
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=1, ncols=1)

        # Create the dump dir if it doesn't exist
        os.makedirs(dump_dir, exist_ok=True)

        # Load the data
        with open(os.path.join(data_dir, 'marker_corners.pickle'), 'rb') as f:
            self.corners = pickle.load(f)
        with open(os.path.join(data_dir, 'marker_ids.pickle'), 'rb') as f:
            self.ids = pickle.load(f)
        with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
            self.commands = pickle.load(f)
        
        print(f'len(self.ids): {len(self.ids)}, len(self.commands): {len(self.commands)}')

        self.camera_intrinsics = np.array([[612.82019043,   0.        , 322.14050293],
                              [  0.        , 611.48303223, 247.9083252 ],
                              [  0.        ,   0.        ,   1.        ]])
        self.distortion_coefficients = np.zeros((5))

        # print(self.corners[0]) # First element of corners will always be non None

        # NOTE: ID Checking should be made
        # Averaging corners 
        i = 0
        j = 1
        while j < len(self.corners):
            while j < len(self.corners)-1 and len(self.corners[j]) == 0:
                j += 1
            
            print('i: {}, j: {}'.format(i, j))
            print('self.ids[i]: {}, self.ids[j]: {}'.format(self.ids[i], self.ids[j]))
            # Traverse from i to j in self.corners and put each step difference
            prev_corners = self.corners[i] # We know that both of them at least have one corner 
            next_corners = self.corners[j] # if one of them has two then second one will be appended to all of them
            interval = j - i
            for curr_cor in range(i+1,j):

                for k in range(min(len(prev_corners), len(next_corners))):
                    curr_step = (next_corners[k][0,:] - prev_corners[k][0,:]) / interval 
                    print('curr_step: {}'.format(curr_step))

                    if self.ids[i][k] == self.ids[j][k]: # If the ids are swapped in the middle that makes no sense 
                        # print(self.corners[curr_cor])
                        self.corners[curr_cor].append(self.corners[curr_cor-1][k] + curr_step)

                if len(prev_corners) > len(next_corners): # If in ith step there were more detected markers just reflect that marker to all interval
                    self.corners[curr_cor].append(prev_corners[-1])
                elif len(next_corners) > len(prev_corners): # If in jth step there were more detected markers
                    self.corners[curr_cor].append(next_corners[-1])

            i = j
            j += 1

        for corner in self.corners:
            print(f'{corner}\n-----')
        
        last_non_empty_corner = None
        min_x, min_y, max_x, max_y = 0,0,0,0
        for i in range(len(self.corners)):
            if len(self.corners[i]) > 0:
                last_non_empty_corner = self.corners[i].copy()
            else:
                self.corners[i] = last_non_empty_corner.copy()

            if min(self.corners[i][0][0,:,0]) < min_x: 
                min_x = min(self.corners[i][0][0,:,0])
            if max(self.corners[i][0][0,:,0]) > max_x: 
                max_x = max(self.corners[i][0][0,:,0])

            if min(self.corners[i][0][0,:,1]) < min_y: 
                min_y = min(self.corners[i][0][0,:,1])
            if max(self.corners[i][0][0,:,1]) > max_y: 
                max_y = max(self.corners[i][0][0,:,1])

        # for i,corners in enumerate(self.corners):
        #     rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.01,
        #                                                                self.camera_intrinsics,
        #                                                                self.distortion_coefficients)
        #     print('corners:\n{}\nrvec:\n{}\ntvec:\n{}\n----'.format(
        #         corners, rvec, tvec
        #     ))

        # Set the axes
        num_frames = len(self.corners)
        self.line, = self.axs.plot([], [])
        self.dir = 0
        self.fps = fps
        # self.pos = np.zeros((len(self.corners), 2)) # This will be filled with forward and side position
        self.axs.set_ylim(min_y, max_y)
        self.axs.set_xlim(min_x, max_x)
        self.axs.set_title("Predicted Markers")

        # Create the animation object and save it
        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = num_frames
        )
        self.anim.save(os.path.join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('Animation saved to: {}'.format(dump_file))

    def init_fun(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        # self.axs.patches = []
        # for j in range(len(self.corners[i])):
        #     curr_polygon = self.corners[i][j][0,:]
        #     if j == 0:
        #         p = patches.Polygon(curr_polygon, edgecolor='r', facecolor='none')
        #     else:
        #         p = patches.Polygon(curr_polygon, edgecolor='g', facecolor='none')
        #     self.axs.add_patch(p)

        # Draw two axes, axes are represented by two arrows starting from the middle of the 
        # rectangles and go to the right top and left corners
        self.axs.patches = []
        for j in range(len(self.corners[i])):
            curr_polygon = self.corners[i][j][0,:,:]
            mean_x, mean_y = curr_polygon[:, 0].mean(), curr_polygon[:, 1].mean()
            right_top_x, right_top_y = curr_polygon[0,0], curr_polygon[0,1]
            right_bot_x, right_bot_y = curr_polygon[1,0], curr_polygon[1,1]

            blue_arr = patches.Arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='b')
            red_arr = patches.Arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
            # self.axs.arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='b')
            # self.axs.arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
            self.axs.add_patch(blue_arr)
            self.axs.add_patch(red_arr)

        return self.line,


if __name__ == "__main__":
    demo_name = 'box_marker_35'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests'
    dump_file = '{}_test.mp4'.format(demo_name)
    fps = 15

    AnimateMarkers(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = f'marker_{dump_file}', 
        fps = fps
    )

    AnimatePosFrame(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = f'pos_{dump_file}', 
        fps = fps
    )
