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
from tqdm import tqdm


from cv2 import aruco
from unitree_legged_msgs.msg import HighCmd, HighState
from matplotlib.animation import FuncAnimation, FFMpegWriter
from contrastive_learning.tests.plotting import plot_corners

# TODO: Merge these two animation classes

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
                    with open(os.path.join(data_dir, 'pos_corners.pickle'), 'rb') as f:
                        self.pos_corners = pickle.load(f)
                    with open(os.path.join(data_dir, 'smoothened_corners.npy'), 'rb') as f:
                        self.corners_np = np.load(f) 
                    with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
                        self.commands = pickle.load(f) # Will be used to predict the actions
                else:
                    with open(os.path.join(data_dir, 'pos_corners.pickle'), 'rb') as f:
                        self.pos_corners += pickle.load(f)
                    with open(os.path.join(data_dir, 'smoothened_corners.npy'), 'rb') as f:
                        curr_corner = np.load(f)
                        self.corners_np = np.concatenate((self.corners_np, curr_corner)) 
                    with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
                        self.commands += pickle.load(f) # Will be used to predict the actions
        else:
            with open(os.path.join(data_dir, 'pos_corners.pickle'), 'rb') as f:
                self.pos_corners = pickle.load(f)
            with open(os.path.join(data_dir, 'smoothened_corners.npy'), 'rb') as f:
                self.corners_np = np.load(f) 
            with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
                self.commands = pickle.load(f) # Will be used to predict the actions

        # Get the predicted actions if wanted
        self.show_predicted_action = show_predicted_action
        if show_predicted_action: # TODO: fix this
            with open(os.path.join(data_dir, 'predicted_actions.npy'), 'rb') as f:
                self.predicted_actions = np.load(f)
            # self.pred_dir = 0

         # Get the shape of the image
        first_marker_img_path = os.path.join(data_dir, 'videos/color_images/frame_0001.jpg')
        first_marker_img = cv2.imread(first_marker_img_path)
        # print('first_marker_img.shape: {}'.format(first_marker_img.shape))

        # Set the axes
        print('Starting Animation')
        num_frames = len(self.pos_corners)
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


        # Create the animation object and save it
        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = num_frames
        )
        self.anim.save(os.path.join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('Animation saved to: {}'.format(os.path.join(dump_dir, dump_file)))

    def init_fun(self):
        self.img.set_array(self.blank_image.copy())

    def animate(self, i):
        curr_pos, _, action = self.pos_corners[i] # curr_pos.shape: 8,2
        actions = [action]
        if self.show_predicted_action:
            actions.append(self.predicted_actions[i])

        self.img, _ = plot_corners(
            self.axs,
            curr_pos,
            use_img=True,
            img=self.img,
            plot_action=True,
            actions=actions,
        )

        self.pbar.update(1)
        return self.img,

if __name__ == '__main__':
    demo_name = 'box_marker_35'
    # exp_name = out_dir.split('/')[-1]
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests/animations'
    # dump_file = '{}_{}.mp4'.format(demo_name, exp_name)
    dump_file = '{}_markers.mp4'.format(demo_name)
    fps = 15

    fps = 15
    AnimateMarkers(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = dump_file,
        fps = fps,
        show_predicted_action=False
    )
