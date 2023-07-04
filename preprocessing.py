import numpy as np
import cv2

def old_red_filtering(image_to_filter, setting: dict):
    image_to_filter = cv2.cvtColor(image_to_filter, cv2.COLOR_BGR2HSV)
    lower1 = np.array(setting['lower1'])
    upper1 = np.array(setting['upper1'])
    lower2 = np.array(setting['lower2'])
    upper2 = np.array(setting['upper2'])
    lower_mask = cv2.inRange(image_to_filter, lower1, upper1)
    upper_mask = cv2.inRange(image_to_filter, lower2, upper2)
    full_mask = lower_mask + upper_mask
    # result = cv2.bitwise_and(result, result, mask=full_mask)
    return full_mask

def red_filtering(image_to_filter, setting: dict):
    image_to_filter = cv2.cvtColor(image_to_filter, cv2.COLOR_BGR2HSV)
    lower1 = np.array(setting['lower1'])
    upper1 = np.array(setting['upper1'])
    lower2 = np.array(setting['lower2'])
    upper2 = np.array(setting['upper2'])
    lower_mask = cv2.inRange(image_to_filter, lower1, upper1)
    upper_mask = cv2.inRange(image_to_filter, lower2, upper2)
    full_mask = lower_mask + upper_mask
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (setting['x_size'], setting['y_size'])) 
    foreground = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return foreground

def irregular_shape_detection(image_to_filter, setting: dict):

    blur = cv2.GaussianBlur(image_to_filter, (7, 7), 2)
    h, w = image_to_filter.shape[:2]

    # Morphological gradient

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

    # Binarize gradient

    lower = np.array([0, 0, 0])
    upper = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lower, upper)

    # flood fill from the edges to remove edge 

    for row in range(h):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, w-1] == 255:
           cv2.floodFill(binary, None, (w-1, row), 0)

    for col in range(w):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[h-1, col] == 255:
            cv2.floodFill(binary, None, (col, h-1), 0)

    # cleaning mask

    foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # creating background

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    background = cv2.dilate(foreground, kernel, iterations=3)
    unknown = cv2.subtract(background, foreground)

    # watershed
    markers = cv2.connectedComponents(foreground)[1]
    markers += 1  # Add one to all labels so that background is 1, not 0
    markers[unknown==255] = 0  # mark the region of unknown with zero
    markers = cv2.watershed(image_to_filter, markers)

    # Assign the markers a hue between 0 and 179
    hue_markers = np.uint8(179*np.float32(markers)/np.max(markers))
    blank_channel = 255*np.ones((h, w), dtype=np.uint8)
    marker_img = cv2.merge([hue_markers, blank_channel, blank_channel])
    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_HSV2BGR)  
    
    # Label the original image with the watershed markers

    labeled_img = image_to_filter.copy()
    labeled_img[markers>1] = marker_img[markers>1]  # 1 is background color
    labeled_img = cv2.addWeighted(image_to_filter, 0.5, labeled_img, 0.5, 0)

    return labeled_img

def segmentation_and_cropping(image_to_crop, keypoint_dict, setting: dict):

    try:
        test = int(keypoint_dict[9][0]) + int(keypoint_dict[9][1]) + int(keypoint_dict[10][0]) + int(keypoint_dict[10][1]) \
            + int(keypoint_dict[11][0]) + int(keypoint_dict[11][1]) + int(keypoint_dict[12][0]) + int(keypoint_dict[12][1]) 
    except:
        return []

    # estraggo le coordinate dei polsi...
    left_wrist = [int(keypoint_dict[9][0]), int(keypoint_dict[9][1])]
    right_wrist = [int(keypoint_dict[10][0]), int(keypoint_dict[10][1])]
    
    left_wrist_x = int(keypoint_dict[9][0])
    right_wrist_x = int(keypoint_dict[10][0])

    left_wrist_y = int(keypoint_dict[9][1])
    right_wrist_y = int(keypoint_dict[10][1])

    #  e dei fianchi...
    left_hip = [int(keypoint_dict[11][0]), int(keypoint_dict[11][1])]
    right_hip = [int(keypoint_dict[12][0]), int(keypoint_dict[12][1])]

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

def old_segmentation_and_cropping(image_to_crop, keypoint_dict, setting: dict):
    # creo una ROI a partire dalla maschera 
    x, y, w, h = cv2.boundingRect(full_mask)
    rectangle_image = image_to_crop.copy()

    # margine extra a sinistra e destra
    # no cropping su altezza
    width_margin = setting['width_margin']
    height_margin = setting['height_margin']
    # y = 0
    # h = image_to_crop.shape[0]
    # creo un rettangolo a partire dalla ROI
    cv2.rectangle(rectangle_image, (x,y), (x+w+width_margin, y+h+height_margin), (0, 0, 255), 5)
    # croppo l'immagine usando la ROI
    # if (x - extra_margin) < 0 and (x + w + extra_margin) < image_to_crop.shape[1]:
    #     cropped_image = rectangle_image[y:y + h, 0:(x + w + extra_margin)]
    # elif (x - extra_margin) > 0 and (x + w + extra_margin) > image_to_crop.shape[1]:
    #     cropped_image = rectangle_image[y:y + h, (x - extra_margin):image_to_crop.shape[1]]
    # elif (x - extra_margin) < 0 and (x + w + extra_margin) > image_to_crop.shape[1]:
    #     cropped_image = rectangle_image[y:y + h, 0:image_to_crop.shape[1]]
    # else:
    #     cropped_image = rectangle_image[y:y + h, (x - extra_margin):(x + w + extra_margin)]
    # riconverto l'immagine in colori umani
    cropped_image = rectangle_image[ 0 : (y + h + height_margin) , (x - width_margin) : (x + w + width_margin)]
    return cropped_image

def squaring(image_to_square):
    # (H, W, D)
    frame_width = int(image_to_square.shape[1])
    frame_height = int(image_to_square.shape[0])
    pixel_to_rect = int(abs(frame_width - frame_height) / 2)
    squared_image = image_to_square[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
    return cv2.resize(squared_image,(192, 192))