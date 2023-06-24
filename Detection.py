import numpy as np
import cv2
import tensorflow as tf


class Detection:

    def __init__(self, size):
        self.interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.input_size = size
        self.output_details = self.interpreter.get_output_details()

    def excludeKeypoint(self, key_id):
        """
           exclude eyes ear knee ankle
        """
        return key_id != 1 and key_id != 2 and key_id != 3 and key_id != 4 \
            and key_id != 13 and key_id != 14 \
            and key_id != 15 and key_id != 16

    def get_keypoints(self, keypoints_with_scores, width, height, threshold):
        key_id = 0
        keypoint_dict = {}
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            for value in kpts_absolute_xy:
                if (kpts_scores[key_id] > threshold) and self.excludeKeypoint(key_id):
                    keypoint_dict[key_id] = value
                key_id = key_id + 1
            return keypoint_dict

    def draw_keypoint_on_image(self, image, keypoints_with_scores, threshold):

        height, width, channels = image.shape
        keypoint_dict = self.get_keypoints(keypoints_with_scores, width, height, threshold)
        # Test con sfondo nero
        # (192 x 192 x 3) shape immagine di partenza
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        for key in keypoint_dict.keys():
            centro = (int(keypoint_dict[key][0]), int(keypoint_dict[key][1]))
            image = cv2.circle(image, centro, 1, (255, 0, 0), 2)
        return keypoint_dict, image

    def inference(self, image, threshold):
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype=tf.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        keypoint_dict, out_im = self.draw_keypoint_on_image(image, keypoints_with_scores, threshold)
        return keypoint_dict, out_im
