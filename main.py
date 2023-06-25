from Detection import Detection
from Homography import Homography
from preprocessing import red_filtering, segmentation_and_cropping, equalizing, squaring
from tracking import kalman_filter
from HeatMapGenerator import HeatMapGenerator
import cv2
from PIL import Image

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

# dst_point = {0: [410, 292], 1: [422, 281], 2: [396, 281], 3: [444, 287], 4: [372, 287], 5: [483, 362], 6: [342, 362],
#              7: [511, 450], 8: [320, 450], 9: [520, 540], 10: [303, 540], 11: [461, 553], 12: [368, 553],
#              13: [454, 675], 14: [357, 675], 15: [449, 780], 16: [354, 780]}

# webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('video_registrazioni_nao/michael_pushing_free_kick.avi')
# webcam = cv2.VideoCapture('resources/michael_pushing_free_kick.avi')
if not webcam.isOpened():
    raise Exception("Errore nell'apertura della webcam")

input_size = 192
movenet = Detection(input_size)
ret, image = webcam.read()
first_iteration_indicator = 1
hg = None
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
        keypoint_dict, out_im = movenet.inference(image, 0.1)
        out_im = cv2.cvtColor(out_im, cv2.COLOR_BGR2RGB)
        cv2.imshow("Pose estimation", out_im)

        # -----------------------------KALMAN+MUNKRES START HERE-----------------------------
        keypoint_dict = kalman_filter(keypoint_dict)
        # ------------------------------KALMAN+MUNKRES END HERE------------------------------

        if bool(keypoint_dict):
            # -----------------------------HOMOGRAPHY START HERE-----------------------------
            image_pil = Image.fromarray(out_im, 'RGB')
            image_pil.save('model/src.jpg')
            src = out_im
            dst = cv2.imread("model/skeleton_2d.jpg")
            h = Homography(src, dst)
            punti2d = [[410, 292], [483, 362], [342, 362], [461, 553], [368, 553]]
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

            if len(punti2d) < 3 or len(punti3d) < 3:
                pass
            else:
                corr = h.normalize_points(punti2d, punti3d)
                h._compute_view_based_homography(corr)
                plan_view = cv2.warpPerspective(src, h.H, (dst.shape[1], dst.shape[0]))
                cv2.imshow("Pose estimation homography + kalman filter (NO Munkres)", plan_view)
            # ------------------------------HOMOGRAPHY END HERE------------------------------

            # -----------------------------HEATMAP GENERATION START HERE-----------------------------
            # Osserva plan_view è in formato BGR, a causa del metodo warpPerspective di OpenCV
            if first_iteration_indicator == 1:
                hg = HeatMapGenerator()
                hg.generate_heatmap(plan_view, first_iteration_indicator)
                result_overlay = hg.get_result_overlay()
                cv2.imshow("HeatMap", result_overlay)
                first_iteration_indicator = 0
            else:
                hg.generate_heatmap(plan_view, first_iteration_indicator)
                result_overlay = hg.get_result_overlay()
                cv2.imshow("HeatMap", result_overlay)
            # -----------------------------HEATMAP GENERATION END HERE-------------------------------

            # -----------------------------TRACKING START HERE-----------------------------
            # tracked = plan_view.copy()
            # TODO: add tracking code 
            # ------------------------------TRACKING END HERE------------------------------

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    ret, image = webcam.read()

webcam.release()
cv2.destroyAllWindows()
