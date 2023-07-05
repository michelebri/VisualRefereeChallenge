import numpy as np
import cv2

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

def equalizing(image_to_equalize):
    equalized_image = image_to_equalize.copy()
    ycrcb_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2YCR_CB)
    ycrcb_image_channels = cv2.split(ycrcb_image)
    cv2.equalizeHist(ycrcb_image_channels[0], ycrcb_image_channels[0])
    cv2.merge(ycrcb_image_channels, ycrcb_image)
    cv2.cvtColor(ycrcb_image, cv2.COLOR_YCR_CB2BGR, equalized_image)
    return equalized_image

def check_gloves(image_to_check, keypoint_dict):
    try:
        test = int(keypoint_dict[9][0]) + int(keypoint_dict[9][1]) + int(keypoint_dict[10][0]) + int(keypoint_dict[10][1]) 
    except:
        return False

    # estraggo le coordinate dei polsi...
    left_wrist_x = int(keypoint_dict[9][0])
    left_wrist_y = int(keypoint_dict[9][1])

    right_wrist_x = int(keypoint_dict[10][0])
    right_wrist_y = int(keypoint_dict[10][1])

    # disegno un rettangolo solo per capire se sta prendendo i polsi oppure altro, 
    cv2.rectangle(image_to_check, (right_wrist_x, right_wrist_y), (left_wrist_x, left_wrist_y), (0, 0, 255), 20)

    left_wrist_neighbors = np.asarray(get_neighbors(image_to_check, left_wrist_x, left_wrist_y))
    right_wrist_neighbors = np.asarray(get_neighbors(image_to_check, right_wrist_x, right_wrist_y))

    if left_wrist_neighbors.all() > 200 and right_wrist_neighbors.all() > 200:
        return True
    return False

def get_neighbors(matrix, x, y):
    # TODO: da aggiustare perché non funziona
    num_rows, num_cols = len(matrix), len(matrix[0])
    result = []
    
    for i in range( (0 if x-1 < 0 else x-1), (num_rows if x+2 > num_rows else x+2), 1  ):
        for j in range( (0 if y-1 < 0 else y-1), (num_cols if y+2 > num_cols else y+2), 1 ):
            if (matrix[x][y] & matrix[i][j]).any():
                result.append(matrix[i][j])
    return result


def squaring(image_to_square):
    # (H, W, D)
    frame_width = int(image_to_square.shape[1])
    frame_height = int(image_to_square.shape[0])
    pixel_to_rect = int(abs(frame_width - frame_height) / 2)
    squared_image = image_to_square[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
    return cv2.resize(squared_image, (192, 192))


def obstruct_fake_referee(frame, keypoint_dict, setting: dict):
    try:
        test = int(keypoint_dict[10][0]) + int(keypoint_dict[11][1]) + int(keypoint_dict[11][0]) + int(keypoint_dict[12][0]) + int(keypoint_dict[9][0])
    except:
        return frame

    left_wrist_x = int(keypoint_dict[9][0])
    right_wrist_x = int(keypoint_dict[10][0])

    left_hip_x = int(keypoint_dict[11][0])
    left_hip_y = int(keypoint_dict[11][1])
    
    right_hip_x = int(keypoint_dict[12][0])

    max_x = max(left_wrist_x, left_hip_x)
    min_x = min(right_wrist_x, right_hip_x)

    margin = setting['margin']

    cv2.rectangle(img=frame, pt1=(min_x - 50, 0), pt2=(max_x + margin, left_hip_y + margin), color=(0, 0, 0), thickness=-1)
    # frame[0 : frame.shape[0], right_hip_x: left_hip_x] = np.ones(shape=(frame.shape[0] - 0, left_hip_x + right_hip_x, 3))
    return frame