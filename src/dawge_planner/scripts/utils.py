
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

import glob
import os

def fix_frame_naming(dir_name):
    # Gets the directory name, traverses all files there and rename them by filling the 
    # zeros in each frame name
    img_names = glob.glob('{}/frame_*'.format(dir_name))
    img_names = sorted(img_names)

    print(img_names)

    for i in range(len(img_names)):
        frame_name = 'frame_{:04d}.jpg'.format(i)
        new_img_name = os.path.join(dir_name, frame_name)
        # print('new_img_name: {}'.format(new_img_name))
        os.rename(img_names[i], new_img_name)

def return_images_to_video(fps, imgs_path, video_path):
    os.system('ffmpeg -r {} -f image2 -i {}/%*.jpg -vcodec libx264 -profile:v high444 -pix_fmt yuv420p {}'.format(
        fps, # fps
        imgs_path,
        video_path
    ))

def upload_files_to_drive(file_path, title, upload_path=None):
    gauth = GoogleAuth()
    
    # Creates local webserver and auto
    # handles authentication.
    gauth.LocalWebserverAuth()       
    drive = GoogleDrive(gauth)
    
    f = drive.CreateFile({'title': title})
    f.SetContentFile(file_path)
    f.Upload()

    f = None

if __name__ == "__main__":
    data_dir = '/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_a_6'
    color_dir_name = os.path.join(data_dir, 'videos/color_images')
    color_video_path = os.path.join(data_dir, 'videos/color_video.mp4')
    depth_dir_name = os.path.join(data_dir, 'videos/depth_images')
    depth_video_path = os.path.join(data_dir, 'videos/depth_video.mp4')

    # fix_frame_naming(color_dir_name)
    # return_images_to_video(15, color_dir_name, color_video_path)

    upload_files_to_drive(file_path=color_video_path, title='box_a_6_color.mp4', )

