import cv2
import glob
import os
import pickle

from tqdm import tqdm

# Methods to preprocess data - is used in dataset

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root : str, video_type='color') -> None:
    # Convert the video into image sequences and name them with the frames

    video_path = os.path.join(root, f'videos/{video_type}_video.mp4') # TODO: this will be taken from cfg.data_dir
    images_path = os.path.join(root, f'{video_type}_images')
    if os.path.exists(images_path):
        print(f'{images_path} exists dump_video_to_images exiting')
        return
    os.makedirs(images_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    print(f'dumping video in {root}')
    pbar = tqdm(total = frame_count)
    while success: # The matching 
        pbar.update(1)
        cv2.imwrite('{}.png'.format(os.path.join(images_path, 'frame_{}'.format(str(frame_id).zfill(5)))), image)
        success, image = vidcap.read()
        frame_id += 1

    print(f'dumping finished in {root}')

def create_pos_pairs(root : str, frame_interval : int, video_type='color') -> None:
    images_folder = os.path.join(root, f'{video_type}_images')
    image_names = glob.glob(os.path.join(images_folder, 'frame*'))
    image_names = sorted(image_names)

    # TODO: get actions correctly - currently we're not even adding actions there
    pos_pairs = [] # NOTE: this is only images and the next frame with the frame_interval
    for i in range(len(image_names)-frame_interval):
        pos_pairs.append((
            image_names[i],
            image_names[i+frame_interval]
        ))

    with open(os.path.join(root, f'{video_type}_pos_pairs.pkl'), 'wb') as f:
        pickle.dump(pos_pairs, f) # These pos_pairs files are used in dataset