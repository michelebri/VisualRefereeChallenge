from Detection import Detection
from Homography import Homography
from Filter import KalmanWrapper
from postprocessing import segmentation_and_cropping, squaring, obstruct_fake_referee, check_gloves, equalizing
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import yaml

# Movenet Keypoints Output
#
# 0:' nose',
# 1:'left_eye',         2:'right_eye',
# 3: 'left_ear',        4: 'right_ear',
# 5: 'left_shoulder',   6: 'right_shoulder',
# 7: 'left_elbow',      8: 'right_elbow',
# 9: 'left_wrist',      10: 'right_wrist',
# 11: 'left_hip',       12: 'right_hip',
# 13: 'left_knee',      14: 'right_knee',
# 15: 'left_ankle',     16: 'right_ankle'

# dst_point = {0: [676, 296], 1: [687, 282], 2: [661, 282], 3: [713, 288], 4: [638, 288], 5: [750, 367], 6: [607, 367],
#              7: [780, 460], 8: [582, 460], 9: [789, 552], 10: [565, 552], 11: [728, 566], 12: [633, 566],
#              13: [719, 690], 14: [622, 690], 15: [715, 800], 16: [616, 800]}

with open('config.new.yaml', 'r') as file:
    config = yaml.safe_load(file)

frameacq = 0
input_size = 192
frames = 0
indi = 0
movenet = Detection(input_size)

# video = cv2.VideoCapture("video/first.mp4")
video = cv2.VideoCapture("video/second.mp4")
# video = cv2.VideoCapture("video/third.mp4")

if not video.isOpened():
    raise Exception('Errore apertura video')


ret, frame = video.read()

while ret:
    
    if ret:
        image = cv2.flip(frame, 1)
        squared_image = squaring(image)
        cv2.imshow('squared image', squared_image)

        if (squared_image.shape[0] != 0) and (squared_image.shape[1] != 0) and (squared_image.shape[2] != 0):
            
            keypoint_dict, out_im = movenet.inference(squared_image, config['threshold'])
            cv2.imshow('out image', out_im)

            cropped_image = segmentation_and_cropping(squared_image, keypoint_dict, setting=config['crop'])
            cv2.imshow('cropped image', cropped_image)

            equalized_image = equalizing(cropped_image)
            cv2.imshow('equalized image', equalized_image)

            while check_gloves(equalized_image, keypoint_dict) == True:
                # do HOMOGRAPHY
                print('gloves found successfully')
            else:
                obscured_image = obstruct_fake_referee(squared_image, keypoint_dict, setting=config['obstruct'])
                cv2.imshow('obstruct image', obscured_image)

                keypoint_dict, out_im = movenet.inference(obscured_image, config['threshold'])
                cv2.imshow('re-out image', out_im)

                cropped_image = segmentation_and_cropping(obscured_image, keypoint_dict, setting=config['crop'])
                cv2.imshow('re-cropped image', cropped_image)

                equalized_image = equalizing(cropped_image)

        key=cv2.waitKey(1) & 0xFF
        if key==ord("q"):
            break
    ret, frame = video.read()
video.release()