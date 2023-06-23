from Detection import Detection
from Homography import Homography
import numpy as np
import cv2
import time

input_size = 192
movenet = Detection(input_size)
h = Homography()
dst_point = {0: [158, 43], 6: [98, 110], 5: [235, 109], 11: [120, 296], 12: [213, 301]}
movenet_point = []
virtual_point = []
dst = cv2.imread("model/skeleton_2d.jpg")
dst_h, dst_w, _ = dst.shape

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise Exception("Errore nell'apertura della webcam")

frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
pixel_to_rect = int((frame_width - frame_height) / 2)

video_writer = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'mp4v'), 60, (input_size, input_size))

ret, image = webcam.read()
while ret:
    if ret == True:
        image = image[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
        image = cv2.resize(image, (input_size, input_size))

        keypoint_dict, out_im = movenet.inference(image, 0.4)
        for key in keypoint_dict.keys():
            if key != 9 and key != 10 and key != 7 and key != 8:
                movenet_point.append(keypoint_dict[key])
                virtual_point.append(dst_point[key])
        corr = h.normalize_points(virtual_point, movenet_point)
        h._compute_view_based_homography(corr)
        plan_view = cv2.warpPerspective(out_im, h.H, (dst_w, dst_h))
        cv2.imshow("original", out_im)
        cv2.imshow("image", plan_view)
        cv2.waitKey(1)
        video_writer.write(out_im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    ret, image = webcam.read()

webcam.release()
cv2.destroyAllWindows()
video_writer.release()
