import cv2
import glob
import os
import pickle

from tqdm import tqdm

from unitree_legged_msgs.msg import HighCmd

# Methods to preprocess data - is used in dataset

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root : str, video_type: str ='color') -> None:
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

def create_pos_pairs(data_dir: str,
                     frame_interval: int,
                     get_action_mean_std: bool = False,
                     video_type: str ='color') -> None:

    images_folder = os.path.join(data_dir, f'videos/{video_type}_images')
    image_names = glob.glob(os.path.join(images_folder, 'frame_*.jpg'))
    image_names = sorted(image_names)

    # Get actions as well
    with open(os.path.join(data_dir, 'commands.pickle'), 'rb') as f:
        commands = pickle.load(f)

    print('len(commands): {}, len(image_names): {}'.format(len(commands), len(image_names)))
    assert len(commands) == len(image_names), "Commands and images don't have the same size"

    # NOTE: Sometimes image_names has a larger size by 1 but we can just get the minimum size
    min_size = min(len(commands), len(image_names))

    pos_pairs = [] # NOTE: this is only images and the next frame with the frame_interval
    i = 0
    while True:
        j = i+1 # We will get separate actions
        action = (commands[i].forwardSpeed, commands[i].rotateSpeed)
        while j < min_size-1 and j < i+frame_interval and cmds_are_same(commands[i], commands[j]):
            j += 1
        action_j = (commands[j].forwardSpeed, commands[j].rotateSpeed)
        # print(f'commands[i={i}]: {action} - commands[j={j}]: {action_j}')
        
        pos_pairs.append((
            image_names[i],
            image_names[j],
            action
        ))

        if j == min_size-1:
            break
        i = j

    print(f"Data Dir: {data_dir.split('/')[-1]}, Data Length: {len(pos_pairs)}")

    with open(os.path.join(data_dir, f'{video_type}_pos_pairs.pkl'), 'wb') as f:
        pickle.dump(pos_pairs, f) # These pos_pairs files are used in dataset

def cmds_are_same(cmd_a: HighCmd, cmd_b: HighCmd) -> None: # Gets high level commands and compares forwardSpeed and rotateSpeed
    return cmd_a.forwardSpeed == cmd_b.forwardSpeed and cmd_a.rotateSpeed == cmd_b.rotateSpeed

if __name__ == "__main__":
    data_dir = "/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_a_1"
    data_dirs = glob.glob("/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_a_*")
    data_dirs = sorted(data_dirs)
    print('data_dirs: {}'.format(data_dirs))
    video_type = 'color'

    for data_dir in data_dirs:
        print(data_dir)
        create_pos_pairs(data_dir, video_type, frame_interval=8)

