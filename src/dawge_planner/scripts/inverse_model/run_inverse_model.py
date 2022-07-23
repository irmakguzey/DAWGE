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
from contrastive_learning.tests.test_model import load_lin_model
from contrastive_learning.tests.test_data import plot_corners
from contrastive_learning.datasets.state_dataset import StateDataset
from scripts.tasks.high_lvl_task import HighLevelTask

class RunInverseModel(HighLevelTask):
    def __init__(self, out_dir, data_dir, high_cmd_topic, high_state_topic, rate, color_img_topic, fps=15):
        HighLevelTask.__init__(self, high_cmd_topic, high_state_topic, rate)

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None

        self.video_fps = fps
        demo = data_dir.split('/')[-1]
        self.demo_name = '{}_inverse_model_demo'.format(demo)
        self.out_dir = out_dir
        self.images_dir = os.path.join(out_dir,self.demo_name)
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
        self.cfg.data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/test_demos' # We will use all the demos
        self.cfg.batch_size = 1
        self.dataset = StateDataset(self.cfg)
        
        self.waiting_counter = 0
        self.is_init_var = False

        print('len(dataset): {}'.format(len(self.dataset)))

    # Method to dump all the positions to test_demos
    def dump_all_pos(self):
        self.data_loader = data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4) # In each step next pos will be taken one by one
        # self.iter_data_loader = iter(self.data_loader)
        pbar = tqdm(total=len(self.data_loader))
        all_curr_pos = np.zeros((len(self.dataset), self.cfg.pos_dim*2))
        all_next_pos = np.zeros((len(self.dataset), self.cfg.pos_dim*2))
        bs = self.cfg.batch_size
        for i,batch in enumerate(self.data_loader):
            curr_pos, next_pos, _ = [b.to(self.device) for b in batch]
            print('curr_pos.shape: {}, all_curr_pos[i*bs:(i+1)*bs, :].shape: {}'.format(
                curr_pos.shape, all_curr_pos[i*bs:(i+1)*bs, :].shape
            ))
            all_curr_pos[i*bs:(i+1)*bs, :] = curr_pos.cpu().detach().numpy()
            all_next_pos[i*bs:(i+1)*bs, :] = next_pos.cpu().detach().numpy()
            pbar.update(1)

        with open(os.path.join(self.cfg.data_dir, 'all_curr_pos.npy'), 'wb') as f:
            np.save(f, all_curr_pos)
        with open(os.path.join(self.cfg.data_dir, 'all_next_pos.npy'), 'wb') as f:
            np.save(f, all_next_pos)

    def get_best_next_pos(self, curr_pos): # curr_pos should be normalized
        # Find the closest curr_pos from all_curr_pos.npy
        # Return the corresponding next_pos from the same array
        with open(os.path.join(self.cfg.data_dir, 'all_curr_pos.npy'), 'rb') as f:
            all_curr_pos = np.load(f)
        with open(os.path.join(self.cfg.data_dir, 'all_next_pos.npy'), 'rb') as f:
            all_next_pos = np.load(f)

        dist = np.linalg.norm(all_curr_pos - curr_pos, axis=1)
        closest_pos_id = np.argsort(dist)[0]

        return all_next_pos[closest_pos_id], closest_pos_id
        

    def update_high_cmd(self):

        # try:
        # Get the next state position from the dataset
        # _, next_pos, action = next(self.iter_data_loader)
        # _, next_pos, action = next(iter(self.data_loader))
        # next_pos = next_pos.to(self.device)

        # Normalize the curr_pos
        self.get_corners()
        curr_pos = self.dataset.normalize_corner(self.curr_pos).flatten()
        next_pos, closest_id = self.get_best_next_pos(curr_pos)
        _, _, action = self.dataset.getitem(closest_id)

        curr_pos = torch.unsqueeze(torch.FloatTensor(curr_pos), 0).to(self.device)
        next_pos = torch.unsqueeze(torch.FloatTensor(next_pos), 0).to(self.device)

        print('curr_pos: {}, next_pos: {}'.format(curr_pos, next_pos))
        pred_action = self.lin_model(curr_pos, next_pos)

        # Denormalize the action
        print('pred_action.shape: {}, action.shape: {}'.format(
            pred_action.shape, action.shape
        ))
        pred_action = self.dataset.denormalize_action(pred_action[0].cpu().detach().numpy()) # NOTE: what is the 0 for?
        action = self.dataset.denormalize_action(action.cpu().detach().numpy())
        print('pred_action: {}, action: {}'.format(pred_action, action))

        # Update the high level command
        self.high_cmd_msg.mode = 2
        self.high_cmd_msg.forwardSpeed = pred_action[0]
        self.high_cmd_msg.rotateSpeed = pred_action[1]

        # Plot and publish the positions
        _, frame_axis = plot_corners(
            self.ax, self.curr_pos,
            color_scheme=1)
        # Plot the next_pos
        next_pos = self.dataset.denormalize_corner(next_pos.cpu().detach().numpy())
        _, frame_axis = plot_corners(
            self.ax, next_pos,
            use_frame_axis=True, frame_axis=frame_axis,
            plot_action=True, actions=(action, pred_action),
            color_scheme=2
        )
        self.frame += 1
        cv2.imwrite(os.path.join(self.images_dir, 'frame_{:04d}.png'.format(self.frame)), cv2.cvtColor(frame_axis, cv2.COLOR_RGB2BGR))
        self.pub_marker_image(frame_axis)

        # except StopIteration:

        #     print('Demo Finished - Converting Demo To a Video')
        #     self.convert_to_video()
        
    def pub_marker_image(self, frame_axis):
        # cv2_img = cv2.cvtColor(frame_axis, cv2.COLOR_RGB2BGR)
        self.state_msg = self.cv_bridge.cv2_to_imgmsg(frame_axis, "rgb8")
        self.state_pub.publish(self.state_msg)

    def get_corners(self):
        if self.color_img_msg is None:
            return [], []
        # Detect the markers
        color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "rgb8")
        gray = cv2.cvtColor(color_cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        curr_corners, curr_ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print('curr_corners: {}'.format(curr_corners))

        for i in range(len(curr_corners)): # Number of markers
            if curr_ids[i] == 1:
                self.curr_pos[:4,:] = curr_corners[i][0,:]
            elif curr_ids[i] == 2:
                self.curr_pos[4:,:] = curr_corners[i][0,:]


    def is_initialized(self):
        if not self.is_init_var:
            if self.color_img_msg is None:
                print('color_img_is None')
                return False
            
            self.get_corners()

            if (self.curr_pos == -1).any():
                print('some of the markers are not initialized')
                return False 

            self.is_init_var = True
        
        return self.is_init_var

    def color_img_cb(self, data):
        self.color_img_msg = data

    def waited_enough(self):
        if self.waiting_counter >= 20:
            self.waiting_counter = 0
            return True 
        else:
            self.waiting_counter += 1
            return False

    def convert_to_video(self): 
        before_dumping = datetime.now()

        color_video_name = '{}/{}.mp4'.format(self.out_dir, self.demo_name)
        os.system('ffmpeg -f image2 -r {} -i {}/%*.png -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
            self.video_fps, # fps
            self.images_dir,
            color_video_name
        ))
        # shutil.rmtree(self.color_video_dir, ignore_errors=True)

        after_dumping = datetime.now()
        time_spent = after_dumping - before_dumping
        print('DUMPING DONE in {} minutes\n-------------'.format(time_spent.seconds / 60.))

    
    def end_signal_handler(self, signum, frame):
        self.convert_to_video()
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

    # task.dump_all_pos() # NOTE: delete this afterwards

    task.run()