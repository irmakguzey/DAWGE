import cv2
import glob
import numpy as np
import os
import pickle

from tqdm import tqdm

from unitree_legged_msgs.msg import HighCmd

# Methods to preprocess data - is used in dataset

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root: str, video_type: str ='color') -> None:
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

    # print('len(commands): {}, len(image_names): {}'.format(len(commands), len(image_names)))
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

    # print(f"Data Dir: {data_dir.split('/')[-1]}, Data Length: {len(pos_pairs)}")

    with open(os.path.join(data_dir, f'{video_type}_pos_pairs.pkl'), 'wb') as f:
        pickle.dump(pos_pairs, f) # These pos_pairs files are used in dataset

def cmds_are_same(cmd_a: HighCmd, cmd_b: HighCmd) -> None: # Gets high level commands and compares forwardSpeed and rotateSpeed
    return cmd_a.forwardSpeed == cmd_b.forwardSpeed and cmd_a.rotateSpeed == cmd_b.rotateSpeed

def smoothen_corners(root: str):
    # Load the data
    with open(os.path.join(root, 'marker_corners.pickle'), 'rb') as f:
        corners = pickle.load(f)
    with open(os.path.join(root, 'marker_ids.pickle'), 'rb') as f:
        ids = pickle.load(f)
        
    # Shape: (Frame Length, 2:Box and Dog, 4: 4 corners, 2: [x,y])
    frame_len = len(corners)
    corners_np = np.zeros((frame_len, 2, 4, 2))
    # First dump all the detected markers to corners_np
    # Put -1 if one of them haven't come into the frame yet
    for i in range(frame_len):
        for j in range(len(corners[i])): # Number of markers
            if ids[i][j] == 1: # The id check is done here
                corners_np[i,0,:] = corners[i][j][0,:]
            else:
                corners_np[i,1,:] = corners[i][j][0,:]

    # Put -1 when the dog or the box hasn't come to the frame yet (in the beginning)
    for i in range(frame_len):
        if corners_np[i,1,:].all() == 0: # Only way is the 2nd one is not initialized hence the 1 as the index
            corners_np[i,1,:] = -1
        else:
            break

    # Traverse through the frames and average smoothen the all 0s
    for marker in range(2):
        prev_corner = None
        for i in range(frame_len):
            if corners_np[i,marker,:].all() == 0 and corners_np[i-1,marker,:].all() != 0: # The first 0, create a loop
                # Find the next non zero index
                j = i 
                while j < len(corners_np)-1 and corners_np[j,marker,:].all() == 0:
                    j += 1
                step = (corners_np[j,marker,:] - corners_np[i-1,marker,:]) / (j-(i-1))
                for k in range(i,j+1):
                    if j == len(corners_np) - 1: # All last frames are 0s:
                        corners_np[k,marker,:] = -1
                    else:
                        corners_np[k,marker,:] = corners_np[k-1,marker,:] + step

    # Dump corners_np
    with open(os.path.join(root, 'smoothened_corners.npy'), 'wb') as f:
        np.save(f, corners_np)

# Load corners_np and commands.pickle and dump them in a similar fashion to pos pairs
def dump_pos_corners(root: str):
    pos_corners = [] # Will have curr dog and box, next dog and box pos and action applied between
    with open(os.path.join(root, 'smoothened_corners.npy'), 'rb') as f:
        smth_corners = np.load(f)
    with open(os.path.join(root, 'commands.pickle'), 'rb') as f:
        commands = pickle.load(f)

    print(f'len(corners): {len(smth_corners)}, len(commands): {len(commands)}')

    # Eliminate the indices where values are -1 (they should be ignored)
    # That means one or both of the markers got out of the frame    
    valid_idx = []
    for i,corner in enumerate(smth_corners):
        if not (corner == -1).any():
            valid_idx.append(i)

    for i in valid_idx[:-1]:
        action = (commands[i].forwardSpeed, commands[i].rotateSpeed)
        pos_corners.append((
            np.concatenate((smth_corners[i,0,:], smth_corners[i,1,:])), # Current box and dog position
            np.concatenate((smth_corners[i+1,0,:], smth_corners[i+1,1,:])), # Next box and dog position
            action # action
        ))

    with open(os.path.join(root, 'pos_corners.pickle'), 'wb') as f:
        pickle.dump(pos_corners, f)


if __name__ == "__main__":
    data_dir = "/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_marker_10"
    data_dirs = glob.glob("/home/irmak/Workspace/DAWGE/src/dawge_planner/data/box_marker_*")
    data_dirs = sorted(data_dirs)
    # print('data_dirs: {}'.format(data_dirs))
    # video_type = 'color'

    for data_dir in data_dirs:
        smoothen_corners(data_dir)
        dump_pos_corners(data_dir)

    # for data_dir in data_dirs:
    #     print(data_dir)
    #     create_pos_pairs(data_dir, video_type, frame_interval=8)
    # smoothen_corners(data_dir)
    # dump_pos_corners(data_dir) 

    

