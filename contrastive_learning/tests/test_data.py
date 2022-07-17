# Script to test the command files that are saved

import cv2
import glob
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

from contrastive_learning.tests.animate_markers import AnimateMarkers

if __name__ == "__main__":
    demo_name = 'box_marker_35'
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/{}'.format(demo_name)
    dump_dir = '/home/irmak/Workspace/DAWGE/contrastive_learning/tests/animations'
    dump_file = '{}_test.mp4'.format(demo_name)

    # data_dirs = glob.glob("/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_marker_*")
    # dump_file_mult = 'all_markers_test.mp4'
    fps = 15

    # AnimateMarkers(
    #     data_dir = data_dirs, 
    #     dump_dir = dump_dir, 
    #     dump_file = dump_file_mult,
    #     fps = fps,
    #     mult_traj = True
    # )

    AnimateMarkers(
        data_dir = data_dir, 
        dump_dir = dump_dir, 
        dump_file = f'marker_{dump_file}', 
        # dump_file = dump_file,
        fps = fps,
        mult_traj = False,
        show_predicted_action = True 
    )

    # AnimatePosFrame(
    #     data_dir = data_dir, 
    #     dump_dir = dump_dir, 
    #     dump_file = f'pos_{dump_file}', 
    #     fps = fps
    # )
