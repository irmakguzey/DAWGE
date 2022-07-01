# Script to test the command files that are saved

import cv2
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

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
        self.pos = np.zeros((num_frames, 2)) # This will be filled with forward and side position
        self.axs.set_ylim(-1, 4)
        self.axs.set_xlim(-4, 4)
        self.axs.set_title("State Position")

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
        self.pos[i][0] = self.states[i].sidePosition
        self.pos[i][1]= self.states[i].forwardPosition

        self.line.set_data(self.pos[:i,0], self.pos[:i,1])

        return self.line,

if __name__ == "__main__":
    demo_name = 'box_a_3'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests'
    dump_file = '{}_test.mp4'.format(demo_name)
    fps = 15

    AnimatePosFrame(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = dump_file, 
        fps = fps
    )
