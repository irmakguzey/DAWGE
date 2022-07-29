#!/usr/bin/env python3

import argparse
import os
import rospy
import signal

from datetime import datetime

from save_stream import SaveStream
from save_robot import SaveRobotCmds
from save_markers import SaveMarkers

class SaveAll:
    def __init__(self, data_dir, fps):
        rospy.init_node('dawge_save', disable_signals=True)

        os.makedirs(data_dir, exist_ok=True)
        video_dir = '{}/videos'.format(data_dir)
        os.makedirs(video_dir, exist_ok=True)

        self.video_saver = SaveStream(
            video_dir=video_dir,
            depth_img_topic='/dawge_camera/depth/image_raw',
            color_img_topic='/dawge_camera/color/image_raw',
            fps=fps
        )      

        self.robot_saver = SaveRobotCmds(
            data_dir=data_dir,
            high_cmd_topic="dawge_high_cmd",
            high_state_topic="dawge_high_state",
            fps=fps
        )

        self.marker_saver = SaveMarkers(
            data_dir=data_dir,
            color_img_topic='/dawge_camera/color/image_raw',
            color_marker_topic='/dawge_camera/color/image_markers',
            fps=fps
        )

        self.frame_count = 0
        self.rate = rospy.Rate(fps)
        signal.signal(signal.SIGINT, self.end_signal_handler)

    def run(self):
        while not rospy.is_shutdown():
            if self.video_saver.initialized() and self.robot_saver.initialized():
                if self.frame_count == 0:
                    print('DATA SAVER INITIALIZED')
                self.frame_count += 1
                self.video_saver.dump_images() # Sends dumps the last received images
                self.robot_saver.append_msgs() # Saves the last received high command
                self.marker_saver.draw_and_dump() # Draws the markers to the current image and dumps them
                print(f'Frame: {self.frame_count}')
            else:
                print(f"Waiting for frames! - video_saver.init: {self.video_saver.initialized()} - robot_saver.init: {self.robot_saver.initialized()}")
            self.rate.sleep()

    def end_signal_handler(self, signum, frame):
        self.video_saver.convert_to_video()
        self.robot_saver.dump()
        self.marker_saver.convert_to_video() # Dump markered video and the corners and ids of the markers
        self.marker_saver.dump_corners()
        rospy.signal_shutdown('Ctrl C pressed')
        exit(1)

if __name__ == "__main__":
    now = datetime.now()
    time_str = now.strftime('%d%m%Y_%H%M%S')

    # Parse arguments - will be used if we want to name the directory ourselves
    parser = argparse.ArgumentParser() 
    parser.add_argument('--save_dir', type=str, default=time_str)

    args = parser.parse_args()

    data_dir = '{}/{}'.format(
        '/home/irmak/Workspace/DAWGE/src/dawge_planner/data',
        args.save_dir
    )

    # TODO: Get the camera fps here
    fps = 15

    data_saver = SaveAll(data_dir, fps)
    data_saver.run()




