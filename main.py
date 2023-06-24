from Detection import Detection
from Homography import Homography
import cv2
from PIL import Image

# Movenet Keypoints Ouput
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

input_size = 192
movenet = Detection(input_size)
dst_point = {0: [160, 42], 1: [171, 31], 2: [145, 31], 3: [194, 34], 4: [121, 34], 5: [235, 111], 6: [93, 111],
             7: [260, 201], 8: [70, 201], 9: [273, 290], 10: [54, 290], 11: [211, 301], 12: [120, 301],
             13: [202, 425], 14: [107, 425], 15: [198, 529], 16: [103, 529]}

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise Exception("Errore nell'apertura della webcam")

frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
pixel_to_rect = int((frame_width - frame_height) / 2)

ret, image = webcam.read()
while ret:
    if ret == True:
        image = image[0:frame_height, pixel_to_rect:(frame_width - pixel_to_rect)]
        image = cv2.resize(image, (input_size, input_size))
        keypoint_dict, out_im = movenet.inference(image, 0.4)
        cv2.imshow("original", out_im)

        if bool(keypoint_dict):
            # -----------------------------HOMOGRAPHY START HERE-----------------------------
            out_im = cv2.cvtColor(out_im, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(out_im, 'RGB')
            image_pil.save('model/src.jpg')
            src = out_im
            dst = cv2.imread("model/skeleton_2d.jpg")
            h = Homography(src, dst)
            punti2d = [[160, 42], [235, 111], [93, 111], [211, 301], [120, 301]]
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
                plan_view = cv2.cvtColor(plan_view, cv2.COLOR_BGR2RGB)
                cv2.imshow("Pose estimation homography", plan_view)
            # ------------------------------HOMOGRAPHY END HERE------------------------------

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    ret, image = webcam.read()

webcam.release()
cv2.destroyAllWindows()
