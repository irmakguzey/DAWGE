#!/usr/bin/env python3

# ROS module to listen to the image topics, draw the detected markers on them and save them in a video

import cv2
import cv_bridge
import rospy
import signal
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('Agg')

from cv2 import aruco
from datetime import datetime

# ROS Image message
from sensor_msgs.msg import Image # Image data will be taken - cv2 bridge can also be used to invert these but for now 

class SaveMarkers: # Class to save image streams
    def __init__(self, data_dir, color_img_topic, color_marker_topic, fps=15):
        '''
        video_dir: directory to dump all the images before converting it to a video - it should be created before
        '''
        # rospy.init_node('dawge_aruco_detector', disable_signals=True) # To convert images to video in the end

        # Create an opencv bridge to save the images
        self.cv_bridge = cv_bridge.CvBridge()
        self.color_img_msg = None

        self.rate = rospy.Rate(fps)
        self.video_fps = fps
        self.data_dir = data_dir
        # Videos dir is already created in save_all
        self.video_dir = os.path.join(data_dir, 'videos') # This is the path where the final videos should be dumped
        self.color_video_dir = os.path.join(data_dir, 'videos/marker_images') # Will be used in the actual dumping
        if os.path.exists(self.color_video_dir):
            shutil.rmtree(self.color_video_dir, ignore_errors=True)
        os.makedirs(self.color_video_dir)

        # Each corner and id should be saved for each frame
        self.corners, self.ids = [], []

        self.camera_intrinsics = np.array([[612.82019043,   0.        , 322.14050293],
                              [  0.        , 611.48303223, 247.9083252 ],
                              [  0.        ,   0.        ,   1.        ]])
        self.distortion_coefficients = np.zeros((5))

        # Initialize ROS listeners
        rospy.Subscriber(color_img_topic, Image, self.color_img_cb)
        # Initialize ROS publisher 
        self.pub = rospy.Publisher(color_marker_topic, Image, queue_size=10)
        self.img_msg = Image()

        signal.signal(signal.SIGINT, self.end_signal_handler) # TODO: not sure what to do here
        self.frame = 0

    def run(self):
        while not rospy.is_shutdown():
            if self.initialized():
                frame_axis = self.draw_markers() # Draws the markers to the current image
                # Publish the new image
                self.pub_marker_image(frame_axis)
                # Dump the images
                self.dump_images()
                self.append_corners()
            else:
                print('Waiting for the images')


    def color_img_cb(self, data): 
        self.color_img_msg = data

    def initialized(self):
        return (self.color_img_msg is not None)

    def draw_markers(self):
        # Detect the markers
        color_cv2_img = self.cv_bridge.imgmsg_to_cv2(self.color_img_msg, "rgb8")
        gray = cv2.cvtColor(color_cv2_img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        self.curr_corners, self.curr_ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print('type(corners): {}'.format(type(self.curr_corners)))
        # Draw markers and axis
        frame_markers = aruco.drawDetectedMarkers(color_cv2_img.copy(), self.curr_corners)
        frame_axis = frame_markers.copy()
        for i in range(len(self.curr_corners)):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(self.curr_corners[i], 0.01,
                                                                    self.camera_intrinsics,
                                                                    self.distortion_coefficients)
            if i == 0:
                frame_axis = aruco.drawAxis(frame_markers.copy(), self.camera_intrinsics, self.distortion_coefficients, rvec, tvec, 0.01)
            else:
                frame_axis = aruco.drawAxis(frame_axis.copy(), self.camera_intrinsics, self.distortion_coefficients, rvec, tvec, 0.01)
        
        # Plot the frame axis
        if self.frame == 0:
            self.img = plt.imshow(frame_axis)
            for i in range(len(self.curr_corners)):
                c = self.curr_corners[i][0]
                self.id_imgs = plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label = "id={0}".format(self.curr_ids[i]))
            plt.legend()
            print('self.id_imgs: {}'.format(self.id_imgs))
        else:
            self.img.set_data(frame_axis)

            for i in range(len(self.curr_corners)):
                c = self.curr_corners[i][0]
                self.id_imgs[0].set_data([c[:, 0].mean()], [c[:, 1].mean()])

        return frame_axis

    def pub_marker_image(self, frame_axis):
        cv2_img = cv2.cvtColor(frame_axis, cv2.COLOR_RGB2BGR)
        self.img_msg = self.cv_bridge.cv2_to_imgmsg(cv2_img, "bgr8")
        self.pub.publish(self.img_msg)

    def dump_corners(self):
        with open('{}/marker_corners.pickle'.format(self.data_dir), 'wb') as pkl:
            pickle.dump(self.corners, pkl, pickle.HIGHEST_PROTOCOL)
        with open('{}/marker_ids.pickle'.format(self.data_dir), 'wb') as pkl:
            pickle.dump(self.ids, pkl, pickle.HIGHEST_PROTOCOL)

    def dump_images(self):
        self.frame += 1
        color_img_path = os.path.join(self.color_video_dir, 'frame_{:04d}.png'.format(self.frame))
        plt.savefig(color_img_path)

    def draw_and_dump(self):
        self.draw_markers()
        self.dump_images()
        self.append_corners()

    def append_corners(self):
        self.corners.append(self.curr_corners)
        self.ids.append(self.curr_ids)

    def convert_to_video(self): 
        before_dumping = datetime.now()

        color_video_name = '{}/markers_video.mp4'.format(self.video_dir)
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
        self.dump_corners()
        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)

if __name__ == "__main__":
    rospy.init_node('dawge_aruco_detector', disable_signals=True)
    aruco_detector = SaveMarkers(
        data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/scripts/aruco_markers/data',
        color_img_topic = '/dawge_camera/color/image_raw',
        color_marker_topic = '/dawge_camera/color/image_markers'
    )
    aruco_detector.run()