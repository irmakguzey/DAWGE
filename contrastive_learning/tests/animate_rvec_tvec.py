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

        # Get the shape of the image
        first_marker_img_path = os.path.join(data_dir, 'videos/color_images/frame_0001.jpg')
        first_marker_img = cv2.imread(first_marker_img_path)
        print('first_marker_img.shape: {}'.format(first_marker_img.shape))

        # Set the axes
        num_frames = len(self.pos_rvec_tvec)
        self.pbar = tqdm(total = num_frames)

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
        
        for j in range(2):
            curr_rvec_tvec = self.pos_rvec_tvec[i][0][j*6:(j+1)*6]
            if j == 0:
                frame_axis = aruco.drawAxis(self.blank_image.copy(),
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

        self.img.set_array(frame_axis)

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
        fps = fps
    )
