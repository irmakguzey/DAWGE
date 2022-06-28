#!/usr/bin/env python3

import os

from datetime import datetime

from save_stream import SaveStream
from save_actions import SaveRobot

if __name__ == "__main__":
    now = datetime.now()
    time_str = now.strftime('%d%m%Y_%H%M%S')
    data_dir = '{}/{}'.format(
        '/home/irmak/Workspace/DAWGE/src/dawge_planner/data',
        time_str
    )
    os.mkdir(data_dir)
    video_dir = '{}/videos'.format(data_dir)

    # TODO: Get the camera fps here
    fps = 30

    # Save the video stream
    video_saver = SaveStream(
        video_dir=video_dir,
        color_img_topic='/dawge_camera/color_image',
        depth_img_topic='/dawge_camera/depth_image',
        cam_fps=fps
    ) 

    video_saver.run()

    # Save the commands and states 
    robot_saver = SaveRobot(
        data_dir=data_dir,
        high_cmd_topic="dawge_high_cmd",
        high_state_topic="dawge_high_state",
        fps=fps
    )

    robot_saver.run()




