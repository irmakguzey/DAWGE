# Script to make keyboard translations to 3 jointed robot arm end effector

import cv2
import os
from os.path import join, dirname, basename
import matplotlib
import math
import glob
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch 
# import torch.nn.functional as F 
# import torch.nn as nn 
# import torch.optim as optim 
import torch.utils.data as data 

from matplotlib.animation import FuncAnimation, FFMpegWriter
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Animation wrapper for plotting actions

class AnimationWrapper:
    def __init__(self, X, Y_pred, Y_act, dump_dir, is_image=False, dump_file='animation.mp4', total_frames=2000, sec_interval=1, fps=20): # directory to dump the animated video
        # if Y is none then poly method will be used to construct Y, else Y will be plotted

        self.X = X # X can be none when there is an image to be given
        self.Y_pred = Y_pred # Y_pred can have multiple images if there are multiple nearest neighbours given
        self.Y_act = Y_act

        self.sec_interval = sec_interval
        self.frame_interval = fps * sec_interval
        self.is_image = is_image # boolean to indicate if the data we're animating is an image or not

        k = Y_pred.shape[1]
        nrows = 2
        # ncols = math.ceil((k+2) / nrows)
        ncols = 3 # NOTE: Simdilik k'yi biliyomus gibi davranicam :D 
        self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=nrows, ncols=ncols) 

        if is_image:
            nrows = 2
            # ncols = math.ceil((k+2) / nrows)
            ncols = 3 # NOTE: Simdilik k'yi biliyomus gibi davranicam :D 
            self.fig, self.axs = plt.subplots(figsize=(15,15), nrows=nrows, ncols=ncols) 

            self.imgs = []
            self.imgs.append([self.axs[0,0].imshow(X[0], cmap='gray')])
            self.imgs[0].append(self.axs[0,1].imshow(Y_act[0], cmap='gray'))
            self.imgs[0].append(self.axs[0,2].imshow(Y_pred[0,0], cmap='gray'))

            self.imgs.append([self.axs[1,0].imshow(Y_pred[0,1], cmap='gray')])
            self.imgs[1].append(self.axs[1,1].imshow(Y_pred[0,2], cmap='gray'))
            self.imgs[1].append(self.axs[1,2].imshow(Y_pred[0,3], cmap='gray'))

            self.axs[0,0].set_title("Current Observation")
            self.axs[0,1].set_title("Next Observation")
            self.axs[0,2].set_title("1st NN")

            self.axs[1,0].set_title("2nd NN")
            self.axs[1,1].set_title("3rd NN")
            self.axs[1,2].set_title("4th NN")

        else:
            self.fig, self.axs = plt.subplots(nrows=1,ncols=2)

            self.line_actual, = self.axs[0].plot([], [])
            self.line_pred, = self.axs[1].plot([], [])
            self.axs[0].set_ylim(-0.5,0.5)
            self.axs[0].set_title("Actual Action")

            self.axs[1].set_ylim(-0.5,0.5)
            self.axs[1].set_title("Predicted Action")

        self.anim = FuncAnimation(
            self.fig, self.animate, init_func = self.init_fun, frames = total_frames
        )

        self.anim.save(join(dump_dir, dump_file), fps=fps, extra_args=['-vcodec', 'libx264'])
        print('animation saved to: {}'.format(join(dump_dir, dump_file)))

    def init_fun(self):
        if self.is_image:
            # self.img_actual.set_array(np.zeros((self.Y_act[0].shape)))
            # self.img_pred.set_array(np.zeros((self.Y_pred[0].shape)))

            # self.imgs.append([self.axs[0,0].imshow(X[0], cmap='gray')])
            # self.imgs[0].append(self.axs[0,1].imshow(Y_act[0], cmap='gray'))
            # self.imgs[0].append(self.axs[0,2].imshow(Y_pred[0,0], cmap='gray'))

            # self.imgs.append([self.axs[1,0].imshow(Y_pred[0,1], cmap='gray')])
            # self.imgs[1].append(self.axs[1,1].imshow(Y_pred[0,2], cmap='gray'))
            # self.imgs[1].append(self.axs[1,2].imshow(Y_pred[0,3], cmap='gray'))

            self.imgs[0][0].set_array(np.zeros((self.X[0].shape)))
            self.imgs[0][1].set_array(np.zeros((self.Y_act[0].shape)))
            self.imgs[0][2].set_array(np.zeros((self.Y_pred[0,0].shape)))

            self.imgs[1][0].set_array(np.zeros((self.Y_pred[0,1].shape)))
            self.imgs[1][1].set_array(np.zeros((self.Y_pred[0,2].shape)))
            self.imgs[1][2].set_array(np.zeros((self.Y_pred[0,3].shape)))

            # return self.img_actual, self.img_pred,
            self.imgs[0][0], self.imgs[0][1], self.imgs[0][2], self.imgs[1][0], self.imgs[1][1], self.imgs[1][2],
        else:
            self.line_actual.set_data([], [])
            self.line_pred.set_data([], [])
            return self.line_actual, self.line_pred,

    def animate(self, i):

        if self.is_image:
            x = self.X[i]
            y_pred = self.Y_pred[i]
            y_act = self.Y_act[i]

            self.imgs[0][0].set_array(x)
            self.imgs[0][1].set_array(y_act)
            self.imgs[0][2].set_array(y_pred[0])

            self.imgs[1][0].set_array(y_pred[1])
            self.imgs[1][1].set_array(y_pred[2])
            self.imgs[1][2].set_array(y_pred[3])

            # self.img_actual.set_array((y_act * 255).astype(np.uint8))
            # self.img_pred.set_array((y_pred * 255).astype(np.uint8))

            return self.imgs[0][0], self.imgs[0][1], self.imgs[0][2], self.imgs[1][0], self.imgs[1][1], self.imgs[1][2],

        else:
            x = self.X[i:i+self.frame_interval]
            y_pred = self.Y_pred[i]
            y_act = self.Y_act[i]

            # print('x: {}, y_pred: {}, y_act: {}'.format(
            #     x.shape, y_pred.shape, y_act.shape
            # ))
            
            self.axs[0].set_xlim(min(x), max(x))
            self.axs[1].set_xlim(min(x), max(x))
            # self.axis.set_ylim(min(y), max(y))

            self.line_actual.set_data(x, y_act)
            self.line_pred.set_data(x, y_pred)

            return self.line_actual, self.line_pred, 

    # AnimationWrapper(
    #     X = X,
    #     Y_pred = Y_pred,
    #     Y_act = Y_act,
    #     is_image=True,
    #     dump_dir = 'animations',
    #     dump_file = dump_file,
    #     total_frames = total_frames,
    #     fps=fps, # which should be quite low
    # )


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Fixing random state for reproducibility
np.random.seed(19680801)


def Gen_RandLine(length, dims=2):
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_RandLine(25, 3) for index in range(50)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=False)

plt.show()