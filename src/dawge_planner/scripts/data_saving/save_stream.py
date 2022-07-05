#!/usr/bin/env python3

# Script to listen to a given image topic and save them as a video
# It saves the frames in a given directory and in the end it converts them to a video

import cv2
import cv_bridge
import rospy
import signal
import os
import shutil

from datetime import datetime

# ROS Image message
from sensor_msgs.msg import Image # Image data will be taken - cv2 bridge can also be used to invert these but for now 

class SaveStream: # Class to save image streams
    def __init__(self, video_dir, color_img_topic, depth_img_topic=None, cam_fps=15):
        '''
        video_dir: directory to dump all the images before converting it to a video - it should be created before
        '''
        # rospy.init_node('dawge_camera_save', disable_signals=True) # To convert images to video in the end

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None
        self.depth_img_msg = None

        self.video_fps = cam_fps
        self.video_dir = video_dir # This is the path where the final videos should be dumped
        self.color_video_dir = os.path.join(video_dir, 'color_images') # Will be used in the actual dumping
        if not os.path.exists(self.color_video_dir):
            os.makedirs(self.color_video_dir) 

        self.depth_video_dir = None
        # Initialize ROS listeners
        rospy.Subscriber(color_img_topic, Image, self.color_img_cb)
        if depth_img_topic is not None:
            rospy.Subscriber(depth_img_topic, Image, self.depth_img_cb)
            self.depth_video_dir = os.path.join(video_dir, 'depth_images')
            if not os.path.exists(self.depth_video_dir):
                os.mkdir(self.depth_video_dir)

        signal.signal(signal.SIGINT, self.end_signal_handler) # TODO: not sure what to do here
        self.frame = 0

    def run(self):
        rospy.spin()

    def color_img_cb(self, data): 
        self.color_img_msg = data

    def depth_img_cb(self, data):
        self.depth_img_msg = data

    def initialized(self):
        return (self.depth_img_msg is not None) and (self.color_img_msg is not None)

    def dump_images(self): # Writes the current image - this is for synchronization
        self.frame += 1
        # stamp = rospy.get_rostime()
        # img_name = float('{}.{:09d}'.format(stamp.secs, stamp.nsecs))

        color_img_path = os.path.join(self.color_video_dir, 'frame_{:04d}.jpg'.format(self.frame))
        depth_img_path = os.path.join(self.depth_video_dir, 'frame_{:04d}.jpg'.format(self.frame))

        color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "bgr8") # It published as bgr8 so the saving should be bgr to make it rgb again
        depth_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.depth_img_msg)

        cv2.imwrite(color_img_path, color_cv2_img)
        cv2.imwrite(depth_img_path, depth_cv2_img)
        print('color img -> {}'.format(color_img_path))

    def convert_to_video(self): 
        before_dumping = datetime.now()

        print('DUMPING VIDEO')
        color_video_name = '{}/color_video.mp4'.format(self.video_dir)
        os.system('ffmpeg -f image2 -r {} -i {}/%*.jpg -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
            self.video_fps, # fps
            self.color_video_dir,
            color_video_name
        ))
        # shutil.rmtree(self.color_video_dir, ignore_errors=True)

        if self.depth_video_dir is not None:
            depth_video_name = '{}/depth_video.mp4'.format(self.video_dir)
            os.system('ffmpeg -f image2 -r {} -i {}/%*.jpg -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
                self.video_fps, # fps
                self.depth_video_dir,
                depth_video_name
            ))
            # shutil.rmtree(self.depth_video_dir, ignore_errors=True)

        after_dumping = datetime.now()
        time_spent = after_dumping - before_dumping
        print('DUMPING DONE in {} minutes\n-------------'.format(time_spent.seconds / 60.))

    def end_signal_handler(self, signum, frame):
        self.convert_to_video()
        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)
        
if __name__ == '__main__':
    
    now = datetime.now()
    time_str = now.strftime('%d%m%Y_%H%M%S')
    data_dir = '{}/{}'.format(
        '/home/irmak/Workspace/DAWGE/src/dawge_planner/data',
        time_str
    )
    os.mkdir(data_dir)
    video_dir = '{}/videos'.format(data_dir)
    # os.mkdir(video_dir)

    data_saver = SaveStream(
        video_dir=video_dir,
        # color_img_topic='/dawge_camera/color_image',
        # depth_img_topic='/dawge_camera/depth_image'
        depth_img_topic='/camera/depth/image_rect_raw',
        color_img_topic='/camera/color/image_raw',
        cam_fps=15
    ) 

    data_saver.run()

