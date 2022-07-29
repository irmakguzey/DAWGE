#!/usr/bin/env python3

# This script will run inside the robot - saving the stream will run in bangalore

# Standard imports
import cv2
import numpy as np

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Realsense imports
import pyrealsense2 as rs

class RealSenseStream(object):
    def __init__(self, cam_serial_num, resolution, cam_fps = 30):
        # Initializing ROS Node
        rospy.init_node('dawge_camera_stream')

        # Disabling scientific notations
        np.set_printoptions(suppress=True)

        # Creating ROS Publishers
        self.color_image_publisher = rospy.Publisher('/dawge_camera/color/image_raw', Image, queue_size = 1)
        self.depth_image_publisher = rospy.Publisher('/dawge_camera/depth/image_raw', Image, queue_size = 1)

        # Initializing CvBridge
        self.bridge = CvBridge()

        # Setting ROS frequency
        self.rate = rospy.Rate(cam_fps)

        # Starting the realsense camera stream
        self._start_realsense(cam_serial_num, resolution[0], resolution[1], cam_fps)
        print("Started the Realsense pipeline for camera: {}!".format(cam_serial_num))

    def _start_realsense(self, cam_serial_num, width, height, fps):
        config = rs.config()
        pipeline = rs.pipeline()
        config.enable_device(cam_serial_num)

        # Enabling camera streams
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # TODO: Fix this - maybe it will be fixed with USB3 cable but rn with these configs we
        # are always getting the error that requests cannot be resolved

        # # Starting the pipeline
        cfg = pipeline.start(config)
        device = cfg.get_device()

        # Setting the depth mode to high accuracy mode
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 2) # High accuracy post-processing mode
        self.realsense = pipeline

        # Obtaining the color intrinsics matrix for aligning the color and depth images
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy], 
            [0, 0, 1]
        ])
        print('intrinsics: {}'.format(intrinsics))

        # Align function - aligns other frames with the color frame
        self.align = rs.align(rs.stream.color)

    def _publish_color_image(self, color_image):
        try:
            color_image = self.bridge.cv2_to_imgmsg(color_image, "rgb8")
        except CvBridgeError as e:
            print(e)

        self.color_image_publisher.publish(color_image)

    def _publish_depth_image(self, depth_image):
        try:
            depth_image = self.bridge.cv2_to_imgmsg(depth_image)
        except CvBridgeError as e:
            print(e)

        self.depth_image_publisher.publish(depth_image)

    def get_rgb_depth_images(self):
        frames = None

        while frames is None:
            # Obtaining and aligning the frames
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Getting the images from the frames
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def stream(self):
        print("Starting stream!\n")

        while True:
            color_image, depth_image = self.get_rgb_depth_images()

            # Publishing the original color and depth images
            self._publish_color_image(color_image)
            self._publish_depth_image(depth_image)

            self.rate.sleep()

if __name__ == "__main__":
    # cam_serial_num = "109622072273" # This is the one on the robot
    cam_serial_num = "023422073116" # The one on bangalore
    height, width = (720, 1280) # TODO: these are not being used rn - you might wanna fix these
    print("Starting to setup camera: {}.".format(cam_serial_num))
    camera = RealSenseStream(
        cam_serial_num = cam_serial_num,
        resolution = (width, height),
        cam_fps = 30
    )
    camera.stream()