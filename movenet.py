
import numpy as np
import cv2
import tensorflow as tf



def get_keypoints(keypoints_with_scores,width,height,threshold):
  keypoints_all = []  
  id = 0
  keypoint_dict = {}
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    for value in kpts_absolute_xy:
      if(kpts_scores[id]>threshold):
        keypoint_dict[id] = value
      id = id+1
    return keypoint_dict

def draw_keypoint_on_image(image,keypoints_with_scores,threshold):

  height, width, channels = image.shape
  keypoint_dict = get_keypoints(keypoints_with_scores,width,height,threshold)
  for key in keypoint_dict.keys():
    centro = (int(keypoint_dict[key][0]),int(keypoint_dict[key][1]))
    image = cv2.circle(image,centro,1,(255, 0, 0),2)
  return image



import time

inizio = time.time()

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded in ms  : " + str((time.time() - inizio) * 1000))
cap = cv2.VideoCapture(0)

input_size = 192
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_to_rect = int((frame_width - frame_height) / 2)
video_writer = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'mp4v'), 60,(input_size,input_size))
count = 0
tempo_trascorso = 0
ciclo = time.time()
while True:
  inizio = time.time()
  _,image = cap.read()
  image = image[0:frame_height, crop_to_rect:(frame_width-crop_to_rect)]
  image = cv2.resize(image,(input_size,input_size))
  input_image = tf.expand_dims(image, axis=0)
  input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
  input_image = tf.cast(input_image, dtype=tf.uint8)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

  interpreter.invoke()

  keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
  #print("Inference in ms : " + str(1000*(time.time() - inizio)))
  out_im = draw_keypoint_on_image(image, keypoints_with_scores,0.4)
  #print("Draw keypoints in ms : " + str(1000*(time.time() - inizio)))
  
  count = count + 1;
  tempo_per_frame = 1000*(time.time() - inizio);
  tempo_trascorso = tempo_trascorso + tempo_per_frame;
  if(tempo_trascorso >1000):
    print("FPS : " + str(count))
    tempo_trascorso = 0
    count = 0
  cv2.imshow("frame",cv2.resize(out_im,(720,720)))
  cv2.waitKey(1)
  video_writer.write(out_im)
print(time.time() - ciclo)
video_capture.release()
video_writer.release()
