# Script to animate rvec and tvec axes

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
from tqdm import tqdm

from unitree_legged_msgs.msg import HighCmd, HighState
from matplotlib.animation import FuncAnimation, FFMpegWriter
from contrastive_learning.tests.plotting import plot_rvec_tvec

CAMERA_INTRINSICS = np.array([[612.82019043,   0.        , 322.14050293],
                              [  0.        , 611.48303223, 247.9083252 ],
                              [  0.        ,   0.        ,   1.        ]])

class AnimateRvecTvec:
    def __init__(self, data_dir, dump_dir, dump_file, fps, mult_traj=False, show_predicted_action=False):
        # Create the figure to draw
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=1, ncols=1)

        # Create the dump dir if it doesn't exist
        os.makedirs(dump_dir, exist_ok=True)

        with open(os.path.join(data_dir, 'pos_rvec_tvec.pickle'), 'rb') as f:
            self.pos_rvec_tvec = pickle.load(f)

        # Get the predicted actions if wanted
        self.show_predicted_action = show_predicted_action
        if show_predicted_action:
            with open(os.path.join(data_dir, 'predicted_actions.npy'), 'rb') as f:
                self.predicted_actions = np.load(f)
            self.dir = 0 # NOTE: We start the directions from 0 since it doesn't really matter
            self.pred_dir = 0
            self.action_pos = (1100,600)

        # Get the shape of the image
        first_marker_img_path = os.path.join(data_dir, 'videos/color_images/frame_0001.jpg')
        first_marker_img = cv2.imread(first_marker_img_path)
        # print('first_marker_img.shape: {}'.format(first_marker_img.shape))

        # Set the axes
        print('Starting Animation')
        num_frames = len(self.pos_rvec_tvec)
        self.pbar = tqdm(total = num_frames)

        # Plot the first image
        self.blank_image = np.ones(first_marker_img.shape, np.uint8) * 255
        self.img = self.axs.imshow(self.blank_image.copy())
        
        self.fps = fps
        self.axs.set_title("Predicted Markers")

        # Create the animation object and save it
        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = num_frames
        )
        self.anim.save(os.path.join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        self.pbar.close()
        print('Animation saved to: {}'.format(os.path.join(dump_dir, dump_file)))

    def init_fun(self):
        self.img.set_array(self.blank_image.copy())

    def animate(self, i):

        # if self.show_predicted_action:
        #     action = self.pos_rvec_tvec[i][2]
        #     pred_action = self.predicted_actions[i]

        # # Draw the axis
        # for j in range(2):
        #     curr_rvec_tvec = self.pos_rvec_tvec[i][0][j*6:(j+1)*6]
        #     if j == 0:
        #         frame_axis = aruco.drawAxis(self.blank_image.copy(),
        #             CAMERA_INTRINSICS,
        #             np.zeros((5)),
        #             curr_rvec_tvec[:3], curr_rvec_tvec[3:],
        #             0.01)
        #     else:
        #         frame_axis = aruco.drawAxis(frame_axis.copy(),
        #             CAMERA_INTRINSICS,
        #             np.zeros((5)),
        #             curr_rvec_tvec[:3], curr_rvec_tvec[3:],
        #             0.01)

        # # Draw the actions
        # if self.show_predicted_action: # Action will be drawn to the bottom right corner
        #     # Actual action
        #     forward_speed = action[0]
        #     rotate_speed = action[1]
        #     self.dir -= rotate_speed
        #     action_x = forward_speed * math.sin(self.dir) * 500 # 250 is only for scaling
        #     action_y = forward_speed * math.cos(self.dir) * 500
        #     # action_arr = patches.Arrow(mean_x, mean_y, -action_x, -action_y, color='c', label='Actual Action') # - is for drawing purposes
        #     frame_axis = cv2.arrowedLine(frame_axis.copy(), self.action_pos,
        #                                  (int(self.action_pos[0]+action_x), int(self.action_pos[1]-action_y)), # Y should be removed from the action
        #                                  color=(0,200,200), thickness=3)

        #     # Predicted action
        #     forward_speed = pred_action[0]
        #     rotate_speed = pred_action[1]
        #     self.pred_dir -= rotate_speed
        #     action_x = forward_speed * math.sin(self.pred_dir) * 500 # 250 is only for scaling
        #     action_y = forward_speed * math.cos(self.pred_dir) * 500
        #     frame_axis = cv2.arrowedLine(frame_axis.copy(), self.action_pos,
        #                                  (int(self.action_pos[0]+action_x), int(self.action_pos[1]-action_y)), # Y should be removed from the action
        #                                  color=(104,43,159), thickness=3)

        # self.img.set_array(frame_axis)



        actions = [self.pos_rvec_tvec[i][2]]
        if self.show_predicted_action:
            actions.append(self.predicted_actions[i])

        self.img, _ = plot_rvec_tvec (
            self.axs,
            self.pos_rvec_tvec[i][0],
            use_img=True,
            img=self.img,
            plot_action=True,
            actions=actions
        )

        self.pbar.update(1)
        return self.img,

if __name__ == "__main__":
    demo_name = 'box_marker_35'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests/animations'
    dump_file = '{}_rvec_tvec.mp4'.format(demo_name)

    fps = 15
    AnimateRvecTvec(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = dump_file,
        fps = fps,
        show_predicted_action=False
    )
