import math
import numpy as np
import cv2


def red_filtering(image_to_filter):
    result = image_to_filter.copy()
    image_to_filter = cv2.cvtColor(image_to_filter, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([5, 255, 255])
    lower2 = np.array([174, 100, 100])
    upper2 = np.array([179, 255, 255])
    lower_mask = cv2.inRange(image_to_filter, lower1, upper1)
    upper_mask = cv2.inRange(image_to_filter, lower2, upper2)
    full_mask = lower_mask + upper_mask
    # result = cv2.bitwise_and(result, result, mask=full_mask)
    return full_mask


def segmentation_and_cropping(image_to_crop, full_mask):
    # creo una ROI a partire dalla maschera 
    x, y, w, h = cv2.boundingRect(full_mask)
    rectangle_image = image_to_crop.copy()
    # margine extra a sinistra e destra
    extra_margin = 100
    # no cropping su altezza
    y = 0
    h = image_to_crop.shape[0]
    # creo un rettangolo a partire dalla ROI
    # cv2.rectangle(rectangle_image, (x,y), (x+w,y+h), (255,0,0), 0)
    # croppo l'immagine usando la ROI
    if (x - extra_margin) < 0 and (x + w + extra_margin) < image_to_crop.shape[1]:
        cropped_image = rectangle_image[y:y + h, 0:(x + w + extra_margin)]
    elif (x - extra_margin) > 0 and (x + w + extra_margin) > image_to_crop.shape[1]:
        cropped_image = rectangle_image[y:y + h, (x - extra_margin):image_to_crop.shape[1]]
    elif (x - extra_margin) < 0 and (x + w + extra_margin) > image_to_crop.shape[1]:
        cropped_image = rectangle_image[y:y + h, 0:image_to_crop.shape[1]]
    else:
        cropped_image = rectangle_image[y:y + h, (x - extra_margin):(x + w + extra_margin)]
    # riconverto l'immagine in colori umani
    return cropped_image


def equalizing(image_to_equalize):
    equalized_image = image_to_equalize.copy()
    ycrcb_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2YCR_CB)
    ycrcb_image_channels = cv2.split(ycrcb_image)
    cv2.equalizeHist(ycrcb_image_channels[0], ycrcb_image_channels[0])
    cv2.merge(ycrcb_image_channels, ycrcb_image)
    cv2.cvtColor(ycrcb_image, cv2.COLOR_YCR_CB2BGR, equalized_image)
    return equalized_image


def squaring(image_to_square):
    # (H, W, D)
    frame_width = int(image_to_square.shape[1])
    frame_height = int(image_to_square.shape[0])
    pixel_to_rect = int(abs(frame_width - frame_height) / 2)
    squared_image = image_to_square[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
    return cv2.resize(squared_image,(192,192))