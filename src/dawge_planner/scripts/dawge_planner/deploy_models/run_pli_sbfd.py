#!/home/irmak/miniconda3/envs/dawge/bin/python

# Script to plot both the inverse and the forward model together
# predict the next state at each time

# As different than run_inverse_model this also predicts the next position when the action is applied 
# also saves the whole trajectory as well

# We will sample action at each step and test the forward model 

from re import L
import cv2
from soupsieve import closest
from tqdm import tqdm
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
from contrastive_learning.tests.test_model import load_sbfd, load_lin_model
from contrastive_learning.tests.plotting import plot_corners
from contrastive_learning.datasets.dataloaders import get_dataloaders
from contrastive_learning.datasets.state_dataset import StateDataset
from dawge_planner.tasks.high_lvl_task import HighLevelTask

# Class that will sample actions and find the trajectory that makes 
class RunInverseForwardModel(HighLevelTask):
    def __init__(self, pli_out_dir, sbfd_out_dir, 
                 video_dump_dir, high_cmd_topic, high_state_topic,
                 rate, color_img_topic, fps=15):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None

        self.video_fps = fps
        now = datetime.now()
        time_str = now.strftime('%d%m%Y_%H%M%S')
        self.demo_name = '{}_inverse_model_demo'.format(time_str)
        self.video_dump_dir = video_dump_dir
        self.images_dir = os.path.join(video_dump_dir, self.demo_name)
        os.makedirs(self.images_dir, exist_ok=True)
        self.frame = 0

        # Each corner and id should be saved for each frame
        self.corners, self.ids = [], []
        self.curr_pos = np.ones((8,2)) * -1 # We are going to fill this up

        # Initialize ROS listeners
        rospy.Subscriber(color_img_topic, Image, self.color_img_cb)

        # Initialize the ROS publisher to plot the action and the position
        self.state_pub = rospy.Publisher('/dawge_curr_state', Image, queue_size=10)
        _, self.ax = plt.subplots(figsize=(50,50), nrows=1, ncols=1)
        self.state_msg = Image()

        signal.signal(signal.SIGINT, self.end_signal_handler) # TODO: not sure what to do here

        # Model based initializations
        # Start the multiprocessing to load the saved models properly
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)
        torch.cuda.set_device(0)

        device = torch.device('cuda:0')

        self.pli_out_dir = pli_out_dir
        self.sbfd_out_dir = sbfd_out_dir
        pli_cfg = OmegaConf.load(os.path.join(pli_out_dir, '.hydra/config.yaml'))
        sbfd_cfg = OmegaConf.load(os.path.join(sbfd_out_dir, '.hydra/config.yaml'))

        print('pli_cfg: {}\nsbfd_cfg: {}'.format(pli_cfg, sbfd_cfg))

        lin_model_path = os.path.join(pli_out_dir, 'models/lin_model.pt')
        pos_encoder_path = os.path.join(sbfd_out_dir, 'models/pos_encoder.pt')
        trans_path = os.path.join(sbfd_out_dir, 'models/trans.pt')

        # Load the position encoder and forward linear model
        if sbfd_cfg.agent.use_encoder == False:
            sbfd_cfg.z_dim = sbfd_cfg.pos_dim*2
        self.pos_encoder, self.trans = load_sbfd(sbfd_cfg, device, pos_encoder_path, trans_path)

        # Load the encoder
        self.lin_model = load_lin_model(pli_cfg, device, lin_model_path)


        # Create a dataset for one trajectory only
        test_traj_path = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/test_demos/box_marker_test_1'
        # Create a dataset with this dir only - sbfd_cfg and pli_cfg differences shouldn't matter in StateDataset
        demo_dataset = StateDataset(pli_cfg, single_dir=True, single_dir_root=test_traj_path)
        demo_loader = data.DataLoader(demo_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.demo_loader_iter = iter(demo_loader)
        _, _, self.all_dset = get_dataloaders(pli_cfg) # all_dset is needed for normalizing and denormalizing positions and actions

        self.waiting_counter = 0
        self.is_init_var = False 

    


