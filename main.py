from Detection import Detection
from Homography import Homography
from preprocessing import red_filtering, segmentation_and_cropping, equalizing, squaring
from HeatMapGenerator import HeatMapGenerator
from Filter import filter, draw_keypoint_on_image
import cv2
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

webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('video_registrazioni_nao/michael_pushing_free_kick.avi')
# webcam = cv2.VideoCapture('resources/michael_pushing_free_kick.avi')
if not webcam.isOpened():
    raise Exception("Errore nell'apertura della webcam")

input_size = 192
movenet = Detection(input_size)
ret, image = webcam.read()
first_iteration_indicator = 1
hg = None
dst = cv2.imread("resources/skeleton_2d.jpg")
h = Homography()
skip = False
inizio = time.time()
frameacq = 0
while ret:

    image = cv2.flip(image, 1)
    # -----------------------------PREPROCESSING START HERE-----------------------------
    # RED FILTERING
    full_mask = red_filtering(image)
    # SEGMENTATION E CROPPING
    cropped_image = segmentation_and_cropping(image, full_mask)
    # NORMALIZATION
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # EQUALIZATION
    equalized_image = equalizing(normalized_image)
    # SQUARING
    squared_image = squaring(equalized_image)
    # ------------------------------PREPROCESSING END HERE------------------------------

    if (squared_image.shape[0] != 0) and (squared_image.shape[1] != 0) and (squared_image.shape[2] != 0):
        # L'algoritmo di cropping se riceve una immagine senza nemmeno un pixel rosso può croppare troppo e
        # annullare una dimensione

        image = cv2.resize(squared_image, (input_size, input_size))
        keypoint_dict, out_im = movenet.inference(image, 0.35)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_BGR2RGB)
        cv2.imshow("Pose estimation", out_im)

        # -----------------------------KALMAN START HERE-----------------------------
        keypoint_dict = filter(keypoint_dict)
        cv2.imshow("More stable keypoint", draw_keypoint_on_image(image.copy(), keypoint_dict))
        # ------------------------------KALMAN END HERE------------------------------

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
                    # plan_view = cv2.warpPerspective(out_im, h.H, (dst.shape[1], dst.shape[0]))
                    # cv2.imshow("Pose estimation homography + kalman filter (NO Munkres)", plan_view)
                    try:
                        test = int(keypoint_dict[9][0]) + int(keypoint_dict[9][1]) + int(keypoint_dict[10][0]) + int(
                            keypoint_dict[10][1])
                    except:
                        skip = True
                    else:
                        # Definisco i punti di origine(polsi) nell'immagine di partenza
                        src_point_1 = np.array([[int(keypoint_dict[9][0]), int(keypoint_dict[9][1])]], dtype=np.float32)
                        src_point_2 = np.array([[int(keypoint_dict[10][0]), int(keypoint_dict[10][1])]], dtype=np.float32)
                        # Eseguo l'omografia sui punti di origine
                        transformed_point_1 = cv2.perspectiveTransform(src_point_1.reshape(-1, 1, 2), h.H)
                        transformed_point_2 = cv2.perspectiveTransform(src_point_2.reshape(-1, 1, 2), h.H)
                        # Disegno il punto trasformato sull'immagine di output (818 x 1047)
                        plan_view = np.zeros((dst.shape[0], dst.shape[1], dst.shape[2]), dtype=np.uint8)
                        cv2.circle(plan_view, (int(transformed_point_1[0][0][0]), int(transformed_point_1[0][0][1])),
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        cv2.circle(plan_view, (int(transformed_point_2[0][0][0]), int(transformed_point_2[0][0][1])),
                                   radius=5, color=(0, 0, 255), thickness=-1)
                        # Visualizzo l'immagine di output
                        cv2.imshow("Pose estimation homography + kalman filter (NO Munkres)", plan_view)
                else:
                    skip = True
                # ------------------------------HOMOGRAPHY END HERE------------------------------
                gesto = "CornerKick"
                # -----------------------------HEATMAP GENERATION START HERE-----------------------------
                # Osserva plan_view è in formato BGR, a causa del metodo warpPerspective di OpenCV
                if first_iteration_indicator == 1 and not skip:
                    hg = HeatMapGenerator()
                    hg.generate_heatmap(plan_view, first_iteration_indicator)
                    result_overlay = hg.get_result_overlay()
                    cv2.imshow("HeatMap_" + gesto, result_overlay)
                    first_iteration_indicator = 0
                    
                elif not skip:
                    hg.generate_heatmap(plan_view, first_iteration_indicator)
                    result_overlay = hg.get_result_overlay()
                    cv2.imshow("HeatMap_nuova acquisizione " + gesto, result_overlay)

                    if time.time() - inizio > 10:
                        result_overlay = cv2.resize(result_overlay, (600, 600))
                        cv2.imwrite("output_heatmap_generator/" + gesto + str(frameacq) + ".jpg", result_overlay)
                        frameacq = frameacq + 1
                        hg.clean()
                        cv2.destroyAllWindows()
                        time.sleep(4)
                        inizio = time.time()
                        first_iteration_indicator = 1


                # -----------------------------HEATMAP GENERATION END HERE-------------------------------

                # -----------------------------TRACKING START HERE-----------------------------
                # tracked = plan_view.copy()
                # TODO: add tracking code
                # ------------------------------TRACKING END HERE------------------------------

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    skip = False
    cv2.imshow('Immagine mint', image)
    ret, image = webcam.read()

webcam.release()
cv2.destroyAllWindows()
