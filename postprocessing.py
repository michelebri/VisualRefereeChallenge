import numpy as np
import cv2

def red_filtering(image_to_filter, setting: dict):
    image_to_filter = cv2.cvtColor(image_to_filter, cv2.COLOR_BGR2HSV)
    lower1 = np.array(setting['lower1'])
    upper1 = np.array(setting['upper1'])
    lower2 = np.array(setting['lower2'])
    upper2 = np.array(setting['upper2'])
    lower_mask = cv2.inRange(image_to_filter, lower1, upper1)
    upper_mask = cv2.inRange(image_to_filter, lower2, upper2)
    full_mask = lower_mask + upper_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (setting['x_size'], setting['y_size'])) 
    foreground = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return foreground

def segmentation_and_cropping(image_to_crop, keypoint_dict, setting: dict):
    try:
        test = int(keypoint_dict[9][0]) + int(keypoint_dict[9][1]) + int(keypoint_dict[10][0]) + int(keypoint_dict[10][1]) \
            + int(keypoint_dict[11][0]) + int(keypoint_dict[11][1]) + int(keypoint_dict[12][0]) + int(keypoint_dict[12][1]) 
    except:
        return image_to_crop

    # estraggo le coordinate dei polsi...
    left_wrist_x = int(keypoint_dict[9][0])
    right_wrist_x = int(keypoint_dict[10][0])

    left_wrist_y = int(keypoint_dict[9][1])
    right_wrist_y = int(keypoint_dict[10][1])

    #  e dei fianchi...
    left_hip_y = int(keypoint_dict[11][1])
    right_hip_y = int(keypoint_dict[12][1])
    max_y = max(left_wrist_y, right_wrist_y, left_hip_y, right_hip_y)

    # margine extra a sinistra e destra
    width_margin = setting['width_margin']
    height_margin = setting['height_margin']

    # sfora in altezza
    if (max_y + height_margin > image_to_crop.shape[0]) and (right_wrist_x - width_margin > 0) and (left_wrist_x + width_margin < image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : image_to_crop.shape[0], (right_wrist_x - width_margin) : (left_wrist_x + width_margin)]
    # sfora in altezza e a sinistra
    elif (max_y + height_margin > image_to_crop.shape[0]) and (right_wrist_x - width_margin < 0) and (left_wrist_x + width_margin < image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : image_to_crop.shape[0], 0 : (left_wrist_x + width_margin)]
    # sfora in altezza, a destra e a sinistra
    elif (max_y + height_margin > image_to_crop.shape[0]) and (right_wrist_x - width_margin < 0) and (left_wrist_x + width_margin > image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : image_to_crop.shape[0], 0 : image_to_crop.shape[1]]
    # sfora in altezza e a destra
    elif (max_y + height_margin > image_to_crop.shape[0]) and (right_wrist_x - width_margin > 0) and (left_wrist_x + width_margin > image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : image_to_crop.shape[0], (right_wrist_x - width_margin) : image_to_crop.shape[1]]
    # sfora a destra
    elif (max_y + height_margin <= image_to_crop.shape[0]) and (right_wrist_x - width_margin > 0) and (left_wrist_x + width_margin > image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : (max_y + height_margin), (right_wrist_x - width_margin) : image_to_crop.shape[1]]
    # sfora a sinistra
    elif (max_y + height_margin <= image_to_crop.shape[0]) and (right_wrist_x - width_margin < 0) and (left_wrist_x + width_margin <= image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : (max_y + height_margin), 0 : (left_wrist_x + width_margin)]
    # sfora a sinistra e destra
    elif (max_y + height_margin <= image_to_crop.shape[0]) and (right_wrist_x - width_margin < 0) and (left_wrist_x + width_margin > image_to_crop.shape[1]):
        cropped_image = image_to_crop[ 0 : (max_y + height_margin), 0 : image_to_crop.shape[1]]
    # non sfora
    else:
        cropped_image = image_to_crop[ 0 : (max_y + height_margin), (right_wrist_x - width_margin) : (left_wrist_x + width_margin)]
    return cropped_image

def squaring(image_to_square):
    # (H, W, D)
    frame_width = int(image_to_square.shape[1])
    frame_height = int(image_to_square.shape[0])
    pixel_to_rect = int(abs(frame_width - frame_height) / 2)
    squared_image = image_to_square[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
    return cv2.resize(squared_image, (192, 192))

def obstruct_fake_referee(frame, keypoint_dict, setting: dict):

    try:
        test = int(keypoint_dict[0][1]) + int(keypoint_dict[9][1]) + int(keypoint_dict[10][0]) + int(keypoint_dict[11][1]) + int(keypoint_dict[11][0]) + int(keypoint_dict[12][0]) + int(keypoint_dict[9][0])
    except:
        return frame

    nose_y = int(keypoint_dict[0][1])
    right_wrist_x = int(keypoint_dict[10][0])
    right_hip_x = int(keypoint_dict[12][0])
    left_hip_x = int(keypoint_dict[11][0])
    left_hip_y = int(keypoint_dict[11][1])
    
    left_wrist_x = int(keypoint_dict[9][0])
    left_wrist_y = int(keypoint_dict[9][1])

    max_x = max(left_wrist_x, left_hip_x)
    min_x = min(right_wrist_x, right_hip_x)

    margin = setting['margin']

    cv2.rectangle(img=frame, pt1=(min_x - 50, 0), pt2=(max_x + margin, left_hip_y + margin), color=(0, 0, 0), thickness=-1)
    # frame[0 : frame.shape[0], right_hip_x: left_hip_x] = np.ones(shape=(frame.shape[0] - 0, left_hip_x + right_hip_x, 3))
    return frame