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
