import numpy as np
import os
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from PIL import Image
from object_detection.utils import ops as utils_ops
from time import sleep
import pyttsx3
from playsound import playsound
from gtts import gTTS



tts = pyttsx3.Engine()

##def talking_tom(text):
##    tts.say(text)
##    tts.runAndWait()
import serial
ser = serial.Serial("COM5", baudrate = '9600',timeout = 0.5)

global a2

sys.path.append("..")


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

def camera():
  global a2
  from utils import label_map_util
  from utils import visualization_utils as vis_util

  MODEL_NAME = 'inference_graph'
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
  PATH_TO_LABELS = 'training/labelmap.pbtxt'

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  def run_inference_for_single_image(image, graph):
      if 'detection_masks' in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
          # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image.shape[0], image.shape[1])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                              feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      global a2
      if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
      if output_dict['detection_classes'][0] == 1 and  output_dict['detection_scores'][0] > 0.70: 
        print('EYE DOWN')
##        sms()
##        text='I NEED FOR FOOD'
##        talking_tom(text)
        ser.write('2'.encode())
        sleep(10)
        ser.write('5'.encode())
        #a2=1
      if output_dict['detection_classes'][0] == 2 and  output_dict['detection_scores'][0] > 0.70 :
        print('EYE LEFT')
        
##        sms()
##        text='I NEED FOR WATER'
##        talking_tom(text)
        ser.write('4'.encode())
        sleep(10)
        ser.write('5'.encode())
        #a2=1
      if output_dict['detection_classes'][0] == 3 and  output_dict['detection_scores'][0] > 0.70:
        print('EYE RIGHT')
##        sms()
##        text='I WANT TO GO TO RESTROOM'
##        talking_tom(text)
        ser.write('3'.encode())
        sleep(10)
        ser.write('5'.encode())
      if output_dict['detection_classes'][0] == 4 and  output_dict['detection_scores'][0] > 0.70:
        print('EYE UP')
        ser.write('1'.encode())
        sleep(10)
        ser.write('5'.encode())
##        sms()
##        text='I WANT TO GO TO BEDROOM'
##        talking_tom(text)
        
##        sleep(15)
        #a2=1
      if a2==1:
        a2=0
        sleep(1)
  ##      email()
        sleep(1)
      return output_dict
  global a2
  a1=0
  a2=0
  import cv2
  cap = cv2.VideoCapture(0)
  try:
      with detection_graph.as_default():
          with tf.Session() as sess:
                  # Get handles to input and output tensors
                  ops = tf.get_default_graph().get_operations()
                  all_tensor_names = {output.name for op in ops for output in op.outputs}
                  tensor_dict = {}
                  for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                  ]:
                      tensor_name = key + ':0'
                      if tensor_name in all_tensor_names:
                          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

                  while True:
      
                      (__, image_np) = cap.read()
                      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                      image_np_expanded = np.expand_dims(image_np, axis=0)
                      cv2.imwrite('capture.jpg',image_np)
                      # Actual detection.
                      output_dict = run_inference_for_single_image(image_np, detection_graph)
                      # Visualization of the results of a detection.
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          output_dict['detection_boxes'],
                          output_dict['detection_classes'],
                          output_dict['detection_scores'],
                          category_index,
                          instance_masks=output_dict.get('detection_masks'),
                          use_normalized_coordinates=True,
                          line_thickness=8)
                      cv2.imshow('object_detection', cv2.resize(image_np,(800,600)))
                      if cv2.waitKey(1)& 0xFF == ord('q'):
                          cap.release()
                          cv2.destroyAllWindows()
                          break
  except Exception as e:
      print(e)
      #cap.release()

##ser.flushInput()
##while True:
##    a=ser.readline().decode('ascii') # reading serial data
##    print(a)
##    b=a
##    for letter in b:
##        if(letter == 's'):
##            print("Run camera code")
camera()
##
##        else:
##          print("Waiting....")
