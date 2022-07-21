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

class AnimateMarkers:
    def __init__(self, data_dir, dump_dir, dump_file, fps, mult_traj=False, show_predicted_action=False):
        # Create the figure to draw
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=1, ncols=1)

        # Create the dump dir if it doesn't exist
        os.makedirs(dump_dir, exist_ok=True)

        # When mult_traj is set to true data_dir is considered as multiple directories 
        # and smoothened_corners in those multiple directories are concatenated
        if mult_traj:
            for i,root in enumerate(data_dir):
                if i == 0:
                    with open(os.path.join(root, 'smoothened_corners.npy'), 'rb') as f:
                        self.corners_np = np.load(f)
                else:
                    with open(os.path.join(root, 'smoothened_corners.npy'), 'rb') as f:
                        curr_corner = np.load(f)
                        self.corners_np = np.concatenate((self.corners_np, curr_corner))
        else: # TODO: Clean this code
            with open(os.path.join(data_dir, 'pos_corners.pickle'), 'rb') as f:
                self.pos_corners = pickle.load(f)
            with open(os.path.join(data_dir, 'smoothened_corners.npy'), 'rb') as f:
                self.corners_np = np.load(f) 
            with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
                self.commands = pickle.load(f) # Will be used to predict the actions

        # print(self.corners_np.shape)
        
        min_x, max_x = np.min(self.corners_np[:,:,:,0]), np.max(self.corners_np[:,:,:,0])
        min_y, max_y = np.min(self.corners_np[:,:,:,1]), np.max(self.corners_np[:,:,:,1])
        print('min_x: {}, max_x: {}, min_y: {}, max_y: {}'.format(min_x, max_x, min_y, max_y))

        # Set the axes
        num_frames = len(self.pos_corners)
        self.line, = self.axs.plot([], [])
        self.dir_initialized = False # To set the direction as wanted
        self.dir = 0 # Used to draw the action
        self.fps = fps
        self.axs.set_ylim(min_y, max_y)
        self.axs.set_xlim(min_x, max_x)
        self.axs.set_title("Predicted Markers")

        # Get the predicted actions if wanted
        self.show_predicted_action = show_predicted_action
        if show_predicted_action: # TODO: fix this
            with open(os.path.join(data_dir, 'predicted_actions.npy'), 'rb') as f:
                self.predicted_actions = np.load(f)
            self.pred_dir = 0

        # Create the animation object and save it
        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = num_frames
        )
        self.anim.save(os.path.join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('Animation saved to: {}'.format(os.path.join(dump_dir, dump_file)))

    def init_fun(self):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        # Draw two axes, axes are represented by two arrows starting from the middle of the 
        # rectangles and go to the right top and left corners
        for p in list(self.axs.patches):    # note the list!
            p.set_visible(False)
            p.remove()
        # self.axs.clear()
        curr_pos, _, action = self.pos_corners[i] # curr_pos.shape: 8,2
        if self.show_predicted_action:
            pred_action = self.predicted_actions[i]

        # print(curr_pos.shape)
        for j in range(2): # j for each marker
            curr_polygon = curr_pos[j*4:(j+1)*4,:]
            # print('curr_polygon: {}'.format(curr_polygon))
            mean_x, mean_y = curr_polygon[:,0].mean(), curr_polygon[:,1].mean()
            right_top_x, right_top_y = curr_polygon[0,0], curr_polygon[0,1]
            right_bot_x, right_bot_y = curr_polygon[1,0], curr_polygon[1,1]

            if j == 0:
                # blue_arr = patches.Arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='b')
                # red_arr = patches.Arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
                blue_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='b', fill=False)
                # self.axs.add_patch(blue_arr)
                # self.axs.add_patch(red_arr)
                self.axs.add_patch(blue_poly)
            else:
                if self.dir_initialized == False:
                    self.dir_initialized = True
                    # Set the action direction
                    front_x, front_y = ( right_top_x + right_bot_x ) / 2, ( right_top_y + right_bot_y ) / 2
                    self.dir = np.arctan2(front_y-mean_y, front_x-mean_x )
                    if self.show_predicted_action:
                        self.pred_dir = np.arctan2(front_y-mean_y, front_x-mean_x )

                forward_speed = action[0]
                rotate_speed = action[1] / (self.fps)
                self.dir -= rotate_speed
                action_x = forward_speed * math.sin(self.dir) * 250 # 250 is only for scaling
                action_y = forward_speed * math.cos(self.dir) * 250
                action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='c', label='Actual Action') # - is for drawing purposes

                # green_arr = patches.Arrow(mean_x, mean_y, right_top_x-mean_x, right_top_y-mean_y, color='g')
                green_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='g', fill=False)
                # red_arr = patches.Arrow(mean_x, mean_y, right_bot_x-mean_x, right_bot_y-mean_y, color='r')
                # self.axs.add_patch(green_arr)
                # self.axs.add_patch(red_arr)
                self.axs.add_patch(action_arr)
                self.axs.add_patch(green_poly)

                # Show the predicted actions if wanted
                if self.show_predicted_action:
                    forward_speed = pred_action[0]
                    rotate_speed = pred_action[1] / self.fps
                    self.pred_dir -= rotate_speed
                    action_x = forward_speed * math.sin(self.pred_dir) * 250 # 250 is only for scaling
                    action_y = forward_speed * math.cos(self.pred_dir) * 250
                    pred_act_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='m', label='Predicted Action')
                    self.axs.add_patch(pred_act_arr)

        self.axs.legend()

        return self.line,