# Random methods to be used 

import numpy as np
import cv2 
import math

def add_arrow(image, action):
    # action: [forwardSpeed, rotateSpeed]
    # image: cv2 read image


    start_pos = (240, 240) # TODO: Will change this
    # print('action: {}'.format(action))

    # Calculate the length according the linear speed
    # The maximum velocity is 0.2 m/s
    arrow_len = 100 * abs(action[0])/0.6 # TODO: check this again

    # Calculate the ending point
    action_x = math.ceil(arrow_len * math.sin(action[1]))
    action_y = math.ceil(arrow_len * math.cos(action[1]))

    # Signs are for pictures:
    # up means negative in y pixel axis, and right means positive 
    if action[1] > 0:
        end_pos = (start_pos[0] + action_x, start_pos[1] - action_y)
    else:
        end_pos = (start_pos[0] - action_x, start_pos[1] + action_y)

    print(f'(start_pos={start_pos}, end_pos={end_pos})') # TODO: add proper arrow function

    # print('image.shape: {}'.format(image.shape))
    # image = cv2.arrowedLine(image, start_pos, end_pos, (0,255,255), 6)
    max_x, min_x = max(start_pos[0], end_pos[0]), min(start_pos[0], end_pos[0])
    max_y, min_y = max(start_pos[1], end_pos[1]), min(start_pos[1], end_pos[1])
    image[0, :, min_x:max_x, min_y:max_y] = np.zeros((3, max_x-min_x, max_y-min_y))
    # print('image[0, :, start_pos[0]-1:start_pos[0]+1, start_pos[1]-1:start_pos[1]+1].shape: {}'.format(image[0, :, start_pos[0]-1:start_pos[0]+1, start_pos[1]-1:start_pos[1]+1].shape))
    image[0, :, start_pos[0]-1:start_pos[0]+1, start_pos[1]-1:start_pos[1]+1] = np.zeros((3,2,2))

    return image
