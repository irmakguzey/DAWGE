#!/usr/bin/env python3

# ROS module to listen to the image topics, draw the detected markers on them and save them in a video

import cv2
import cv_bridge
import rospy
import signal
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from cv2 import aruco
from datetime import datetime

# ROS Image message
from sensor_msgs.msg import Image # Image data will be taken - cv2 bridge can also be used to invert these but for now 

class ArucoDetector: # Class to save image streams
    def __init__(self, video_dir, color_img_topic, cam_fps=15):
        '''
        video_dir: directory to dump all the images before converting it to a video - it should be created before
        '''
        rospy.init_node('dawge_aruco_detector', disable_signals=True) # To convert images to video in the end

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None

        self.rate = rospy.Rate(cam_fps)
        self.video_fps = cam_fps
        self.video_dir = video_dir # This is the path where the final videos should be dumped
        self.color_video_dir = os.path.join(video_dir, 'color_images') # Will be used in the actual dumping
        if os.path.exists(self.color_video_dir):
            shutil.rmtree(self.color_video_dir, ignore_errors=True)
        os.makedirs(self.color_video_dir)


        # Initialize ROS listeners
        rospy.Subscriber(color_img_topic, Image, self.color_img_cb)

        signal.signal(signal.SIGINT, self.end_signal_handler) # TODO: not sure what to do here
        self.frame = 0

    def run(self):
        while not rospy.is_shutdown():
            if self.initialized():
                self.draw_markers() # Draws the markers to the current image
                # self.dump_images(frame_markers)
            else:
                print('Waiting for the images')


    def color_img_cb(self, data): 
        self.color_img_msg = data

    def initialized(self):
        return (self.color_img_msg is not None)

    def draw_markers(self):
        # Detect and draw the images on the image
        color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "bgr8")
        gray = cv2.cvtColor(color_cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print('corners: {}, ids: {}'.format(corners, ids))
        frame_markers = aruco.drawDetectedMarkers(color_cv2_img.copy(), corners)
        
        # Save the image
        if self.frame == 0:
            self.img = plt.imshow(frame_markers)
        else:
            self.img.set_data(frame_markers)
        # for i in range(len(ids)):
        #     c = corners[i][0]
        #     if self.frame == 0:
        #         selfplt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(ids[i]))
        # plt.legend()
        self.frame += 1
        color_img_path = os.path.join(self.color_video_dir, 'frame_{:04d}.png'.format(self.frame))
        plt.savefig(color_img_path)

    # def dump_images(self, frame_markers): # Writes the current image - this is for synchronization
    #     self.frame += 1
    #     color_img_path = os.path.join(self.color_video_dir, 'frame_{:04d}.jpg'.format(self.frame))
    #     # color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "bgr8") # It published as bgr8 so the saving should be bgr to make it rgb again
    #     cv2.imwrite(color_img_path, frame_markers)
    #     print('color img -> {}'.format(color_img_path))

    def convert_to_video(self): 
        before_dumping = datetime.now()

        color_video_name = '{}/color_video.mp4'.format(self.video_dir)
        os.system('ffmpeg -f image2 -r {} -i {}/%*.png -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
            self.video_fps, # fps
            self.color_video_dir,
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
    aruco_detector = ArucoDetector(
        video_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/scripts/aruco_markers/data',
        color_img_topic = '/dawge_camera/color/image_raw'
    )
    aruco_detector.run()