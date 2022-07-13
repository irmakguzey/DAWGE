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

        # Shape: (Frame Length, 2:Box and Dog, 4: 4 corners, 2: [x,y])
        self.corners_np = np.zeros((len(self.corners), 2, 4, 2))
        # First dump all the detected markers to corners_np
        # Put -1 if one of them haven't come into the frame yet
        for i in range(len(self.corners)):
            for j in range(len(self.corners[i])): # Number of markers
                if self.ids[i][j] == 1: # The id check is done here
                    self.corners_np[i,0,:] = self.corners[i][j][0,:]
                else:
                    self.corners_np[i,1,:] = self.corners[i][j][0,:]


        # Put -1 when the dog or the box hasn't come to the frame yet (in the beginning)
        for i in range(len(self.corners_np)):
            if self.corners_np[i,1,:].all() == 0: # Only way is the 2nd one is not initialized
                self.corners_np[i,1,:] = -1
            else:
                break

        # print('self.corners_np: {}'.format(self.corners_np))

        # Traverse through the frames and average smoothen the all 0s
        for marker in range(2):
            prev_corner = None
            for i in range(len(self.corners_np)):
                if self.corners_np[i,marker,:].all() == 0 and self.corners_np[i-1,marker,:].all() != 0: # The first 0, create a loop
                    # Find the next non zero index
                    j = i 
                    while self.corners_np[j,marker,:].all() == 0:
                        j += 1
                    step = (self.corners_np[j,marker,:] - self.corners_np[i-1,marker,:]) / (j-(i-1))
                    for k in range(i,j):
                        self.corners_np[k,marker,:] = self.corners_np[k-1,marker,:] + step
        
        print(self.corners_np)
        
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

        # Set the axes
        num_frames = len(self.corners)
        self.line, = self.axs.plot([], [])
        self.dir = 0
        self.fps = fps
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
        # Draw two axes, axes are represented by two arrows starting from the middle of the 
        # rectangles and go to the right top and left corners

        self.axs.patches = []
        for j in range(len(self.corners_np[i])):
            curr_polygon = self.corners_np[i,j,:]
            mean_x, mean_y = curr_polygon[:,0].mean(), curr_polygon[:,1].mean()
            right_top_x, right_top_y = curr_polygon[0,0], curr_polygon[0,1]
            right_bot_x, right_bot_y = curr_polygon[1,0], curr_polygon[1,1]

            if j == 0:
                blue_arr = patches.Arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='b')
                red_arr = patches.Arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
                self.axs.add_patch(blue_arr)
                self.axs.add_patch(red_arr)
            else:
                blue_arr = patches.Arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='g')
                red_arr = patches.Arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
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

    # AnimatePosFrame(
    #     data_dir = data_dir, 
    #     dump_dir = dump_dir, 
    #     dump_file = f'pos_{dump_file}', 
    #     fps = fps
    # )
