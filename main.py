from Detection import Detection
from Homography import Homography
from preprocessing import red_filtering, segmentation_and_cropping, squaring
from Filter import KalmanWrapper, draw_keypoint_on_image
import cv2
import matplotlib.pyplot as plt

import os
import numpy as np
import time

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
frameacq = 0
input_size = 192
frames = 0
indi = 0
movenet = Detection(input_size)
dst = cv2.imread("resources/skeleton_2d.jpg")
h = Homography()
kalman_wrapper = KalmanWrapper(max_prediction=5)
kernel = np.array([[1, 2, 2, 1], [2, 6, 6, 2], [2, 6, 6, 2], [1, 2, 2, 1]])
for video in os.listdir("video"):
    frames = 0
    print(video)
    gesto = video.split("_")[0]
    webcam = cv2.VideoCapture("video/" + video)
    frame_count = int(webcam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Numero di frame nel video:", frame_count)
    first_iteration_indicator = 1
    skip = False
    inizio = time.time()
    ret, image = webcam.read()
    heatmapmod = np.zeros((dst.shape[0], dst.shape[1]))
    heatmapmod3 = np.zeros((dst.shape[0], dst.shape[1]))
    heatmapmod5 = np.zeros((dst.shape[0], dst.shape[1]))
    while ret:
        ret, image = webcam.read()
        frames += 1
        if ret:
            image = cv2.flip(image, 1)
            # risparmio secondi 
            # full_mask = red_filtering(image)
            # cropped_image = segmentation_and_cropping(image, full_mask)
            # normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # equalized_image = equalizing(normalized_image)
            squared_image = squaring(image)
            if (squared_image.shape[0] != 0) and (squared_image.shape[1] != 0) and (squared_image.shape[2] != 0):
                keypoint_dict, out_im = movenet.inference(squared_image, 0.35)
                # -----------------------------KALMAN START HERE-----------------------------
                measurement = {}
                for key in keypoint_dict.keys():
                    # extract current measured coords for keypoint labeled 'key'
                    measure = np.array([[keypoint_dict[key][0]], [keypoint_dict[key][1]]])
                    # print("measured coordinate={} for keypoint={}".format(measure, key))
                    # adding measured keypoint coords to measurement dictionary
                    measurement.update({key: measure})
                keypoint_dict = kalman_wrapper.update(measurement)
                # cv2.imshow("More stable keypoint", draw_keypoint_on_image(image.copy(), keypoint_dict))
                # cv2.waitKey(1)
                if bool(keypoint_dict):
                    # -----------------------------HOMOGRAPHY START HERE-----------------------------
                    punti2d = [[676, 296], [750, 367], [607, 367], [728, 566], [633, 566]]
                    punti3d = []
                    index_list = [0, 5, 6, 11, 12]
                    count = 0
                    index_to_remove = []
                    for i in index_list:
                        try:
                            punti3d.append([int(keypoint_dict[i][0]), int(keypoint_dict[i][1])])
                        except:
                            index_to_remove.append(count)
                        finally:
                            count = count + 1
                    for index in sorted(index_to_remove, reverse=True):
                        del punti2d[index]

                    if len(punti2d) < 4 or len(punti3d) < 4:
                        pass
                    else:
                        corr = h.normalize_points(punti2d, punti3d)
                        h._compute_view_based_homography(corr)
                    if h.error < 0.07:
                        try:
                            test = int(keypoint_dict[9][0]) + int(keypoint_dict[9][1]) + int(
                                keypoint_dict[10][0]) + int(keypoint_dict[10][1])
                        except:
                            skip = True
                        else:
                            # Definisco i punti di origine(polsi) nell'immagine di partenza
                            src_point_1 = np.array([int(keypoint_dict[9][0]), int(keypoint_dict[9][1])],
                                                   dtype=np.float32)
                            src_point_2 = np.array([int(keypoint_dict[10][0]), int(keypoint_dict[10][1])],
                                                   dtype=np.float32)
                            # Eseguo l'omografia sui punti di origine
                            transformed_point_1 = cv2.perspectiveTransform(src_point_1.reshape(-1, 1, 2), h.H)
                            transformed_point_2 = cv2.perspectiveTransform(src_point_2.reshape(-1, 1, 2), h.H)
                            x1 = int(transformed_point_1[0][0][1])
                            y1 = int(transformed_point_1[0][0][0])
                            x2 = int(transformed_point_2[0][0][1])
                            y2 = int(transformed_point_2[0][0][0])
                            if frames % 3 == 0:
                                if ((x1 - 2) >= 0 and (x1 + 2) < heatmapmod3.shape[0]
                                        and (y1 - 2) >= 0 and (y1 + 2) < heatmapmod3.shape[1]):
                                    heatmapmod3[x1 - 2:x1 + 2, y1 - 2:y1 + 2] += kernel
                                if ((x2 - 2) >= 0 and (x2 + 2) < heatmapmod3.shape[0]
                                        and (y2 - 2) >= 0 and (y2 + 2) < heatmapmod3.shape[1]):
                                    heatmapmod3[x2 - 2:x2 + 2, y2 - 2:y2 + 2] += kernel
                            if frames % 5 == 0:
                                if ((x1 - 2) >= 0 and (x1 + 2) < heatmapmod5.shape[0]
                                        and (y1 - 2) >= 0 and (y1 + 2) < heatmapmod5.shape[1]):
                                    heatmapmod5[x1 - 2:x1 + 2, y1 - 2:y1 + 2] += kernel
                                if ((x2 - 2) >= 0 and (x2 + 2) < heatmapmod5.shape[0]
                                        and (y2 - 2) >= 0 and (y2 + 2) < heatmapmod5.shape[1]):
                                    heatmapmod5[x2 - 2:x2 + 2, y2 - 2:y2 + 2] += kernel

                            if ((x1 - 2) >= 0 and (x1 + 2) < heatmapmod.shape[0]
                                    and (y1 - 2) >= 0 and (y1 + 2) < heatmapmod.shape[1]):
                                heatmapmod[x1 - 2:x1 + 2, y1 - 2:y1 + 2] += kernel
                            if ((x2 - 2) >= 0 and (x2 + 2) < heatmapmod.shape[0]
                                    and (y2 - 2) >= 0 and (y2 + 2) < heatmapmod.shape[1]):
                                heatmapmod[x2 - 2:x2 + 2, y2 - 2:y2 + 2] += kernel
                    if int(frames / frame_count * 100) == 20:
                        plt.imshow(heatmapmod3, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod3_20.jpg', bbox_inches='tight',
                                    pad_inches=0)
                        plt.imshow(heatmapmod, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_all_20.jpg', bbox_inches='tight',
                                    pad_inches=0)
                        plt.imshow(heatmapmod5, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod5_20.jpg', bbox_inches='tight',
                                    pad_inches=0)
                    if int(frames / frame_count * 100) == 40:
                        plt.imshow(heatmapmod3, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod3_40.jpg', bbox_inches='tight',
                                    pad_inches=0)
                        plt.imshow(heatmapmod, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_all_40.jpg', bbox_inches='tight',
                                    pad_inches=0)
                        plt.imshow(heatmapmod5, cmap='hot', interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod5_40.jpg', bbox_inches='tight',
                                    pad_inches=0)

    plt.imshow(heatmapmod3, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod3_total.jpg', bbox_inches='tight', pad_inches=0)
    plt.imshow(heatmapmod5, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_mod5_total.jpg', bbox_inches='tight', pad_inches=0)
    plt.imshow(heatmapmod, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig('heatmap/' + gesto + "/" + str(indi) + '_all_total.jpg', bbox_inches='tight', pad_inches=0)

    indi = indi + 1;

    plt.close('all')
    print((time.time() - inizio))
