#!/home/irmak/miniconda3/envs/dawge/bin/python
#!/usr/bin/env python3


# Script to load the PLI model that was learned
# At each step we're going to get the position of the robot
# Get the next position in the demo
# Predict the action to be applied from the current position to the next position in the demo
# One demo will be enough for now - then we should do knn matching for the kth state
# and choose the t+1th state in that demo

from re import L
import cv2
import cv_bridge
import rospy
import signal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.utils.data as data 
matplotlib.use('Agg')

from cv2 import aruco
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

# ROS Image message
from sensor_msgs.msg import Image

# Custom imports
from contrastive_learning.tests.test_model import load_lin_model
from contrastive_learning.datasets.state_dataset import StateDataset
from scripts.tasks.high_lvl_task import HighLevelTask

class RunInverseModel(HighLevelTask):
    def __init__(self, out_dir, data_dir, high_cmd_topic, high_state_topic, rate, color_img_topic, fps=15):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None

        self.video_fps = fps
        # self.rate_num = rate

        # Each corner and id should be saved for each frame
        self.corners, self.ids = [], []
        self.curr_pos = np.ones((8,2)) * -1 # We are going to fill this up 

        # Initialize ROS listeners
        rospy.Subscriber(color_img_topic, Image, self.color_img_cb)

        signal.signal(signal.SIGINT, self.end_signal_handler) # TODO: not sure what to do here
        self.frame = 0
        
        print('torch.cuda.is_available(): {}'.format(
            torch.cuda.is_available()
        ))

        # Initialize the process group
        # Start the multiprocessing to load the saved models properly
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29504"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        # Set the parameters for loading the linear model
        self.device = torch.device('cuda:0')
        self.cfg = OmegaConf.load(os.path.join(out_dir, '.hydra/config.yaml'))
        model_path = os.path.join(out_dir, 'models/lin_model.pt')
        # Load the encoder
        self.lin_model = load_lin_model(self.cfg, self.device, model_path)
        
        print('self.lin_model: {}'.format(self.lin_model))

        # Initialize the dataset
        # self.data_dir = data_dir
        # Get the dataset
        self.dataset = StateDataset(self.cfg, single_dir=True, single_dir_root=data_dir)
        self.data_loader = data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4) # In each step next pos will be taken one by one
        self.waiting_counter = 0

        print('len(dataset): {}'.format(len(self.dataset)))

    def update_high_cmd(self):
        # if self.twist_msg is None:
        #     return

        # self.high_cmd_msg.mode = 2

        # # Set the angular velocity - we will use set linear and angular velocity
        # self.high_cmd_msg.rotateSpeed = 0
        # if self.twist_msg.angular.z > 0:
        #     self.high_cmd_msg.rotateSpeed = self.ang_vel 
        # elif self.twist_msg.angular.z < 0:
        #     self.high_cmd_msg.rotateSpeed = -self.ang_vel             

        # # Set the linear velocity 
        # self.high_cmd_msg.forwardSpeed = 0
        # if self.twist_msg.linear.x > 0:
        #     self.high_cmd_msg.forwardSpeed = self.lin_vel 
        # elif self.twist_msg.linear.x < 0:
        #     self.high_cmd_msg.forwardSpeed = -self.lin_vel 

        if self.is_initialized(): # Means that curr_pos has position for both of the markers
            if self.waited_enough(): # Method to increase a counter in this loop so that we can make the robot wait for certain time - will set the limit to the counter by try and fail
                # Get the next state position from the dataset
                _, next_pos, _ = next(iter(self.data_loader))
                next_pos = next_pos.to(self.device)
                # Normalize the curr_pos
                curr_pos = self.dataset.normalize_corner(self.curr_pos).to(self.device)
                curr_pos = torch.unsqueeze(curr_pos, 0)
                # print('curr_pos.shape: {}, next_pos.shape: {}'.format(curr_pos, next_pos))
                pred_action = self.lin_model(curr_pos, next_pos)

                print('pred_action: {}'.format(pred_action))
                # Denormalize the action
                pred_action = self.dataset.denormalize_action(pred_action[0].cpu().detach().numpy()) # NOTE: what is the 0 for?
                print('pred_action denormalized: {}'.format(pred_action))

                # Update the high level command
                self.high_cmd_msg.mode = 2
                self.high_cmd_msg.forwardSpeed = pred_action[0]
                self.high_cmd_msg.rotateSpeed = pred_action[1]

    def get_corners(self):
        if self.color_img_msg is None:
            return [], []
        # Detect the markers
        color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "rgb8")
        gray = cv2.cvtColor(color_cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        curr_corners, curr_ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        for i in range(len(curr_corners)): # Number of markers
            if curr_ids[i] == 1:
                self.curr_pos[i*4:(i+1)*4,:] = curr_corners[i][0,:]

        # print('self.curr_pos: {}'.format(self.curr_pos))


    def is_initialized(self):
        if self.color_img_msg is None:
            print('color_img_is None')
            return False
        
        self.get_corners()

        if (self.curr_pos == -1).any():
            print('some of the markers are not initialized')
            return False 

        return True

    def color_img_cb(self, data):
        self.color_img_msg = data

    def waited_enough(self):
        if self.waiting_counter >= 20:
            self.waiting_counter = 0
            return True 
        else:
            self.waiting_counter += 1
            return False
    
    def end_signal_handler(self, signum, frame):
        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)

if __name__ == "__main__":
    rospy.init_node('dawge_pli', disable_signals=True) # To convert images to video in the end
    
    task = RunInverseModel(
        data_dir='/home/irmak/Workspace/DAWGE/src/dawge_planner/data/test_demos/box_marker_35', # We will use test_demos demos
        out_dir='/home/irmak/Workspace/DAWGE/contrastive_learning/out/2022.07.21/15-35_pli_ue_False_lf_mse_fi_1_pt_corners_bs_64_hd_64_lr_0.001_zd_8',
        high_cmd_topic='dawge_high_cmd',
        high_state_topic='dawge_high_state',
        rate=100, # TODO: Maybe take a look at this?
        color_img_topic='/dawge_camera/color/image_raw',
        fps=15
    )

    task.run()