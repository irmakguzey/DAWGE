# Script to test the command files that are saved

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

# Function to draw box and dog position and applied action
def plot_state(ax, curr_pos, plot_action, action=None, fps=15, color_scheme=1): # Color scheme is to have an alternative color for polygon colors
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
            # Set the action direction
            front_x, front_y = ( right_top_x + right_bot_x ) / 2, ( right_top_y + right_bot_y ) / 2
            curr_dir = np.arctan2(front_y-mean_y, front_x-mean_x )
            
            if plot_action: # Action can be none for the other positions
                # Show the current action 
                forward_speed = action[0]
                rotate_speed = action[1] / (fps)
                curr_dir -= rotate_speed
                action_x = forward_speed * math.sin(curr_dir) * 250 # 250 is only for scaling
                action_y = forward_speed * math.cos(curr_dir) * 250
                action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='c', label='Actual Action') # - is for drawing purposes
                ax.add_patch(action_arr)
            
            # Show the dog position
            if color_scheme == 1:
                dog_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='g', fill=False, label='Dog Position')
            else:
                dog_poly = patches.Polygon(np.concatenate((curr_polygon, curr_polygon[0:1])), color='k', fill=False, label='Predicted Dog Position')
            ax.add_patch(dog_poly)

            ax.plot()
            ax.legend()

if __name__ == "__main__":
    demo_name = 'box_marker_35'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests/animations'
    dump_file = '{}_action_test.mp4'.format(demo_name)

    data_dirs = glob.glob("/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_marker_*")
    dump_file_mult = 'all_markers_test.mp4'
    fps = 15

    AnimateMarkers(
        data_dir = data_dirs, 
        dump_dir = dump_dir, 
        dump_file = dump_file_mult,
        fps = fps,
        mult_traj = True
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
