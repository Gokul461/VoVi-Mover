import urllib.request
import numpy as np
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
import serial
from time import sleep
import cv2

# Initialize serial communication
ser = serial.Serial("COM3", baudrate='9600', timeout=0.5)

# Check TensorFlow version
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Setup TensorFlow model
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'

detection_graph = tf.Graph()kikjijdn
ksmmieci
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
    
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    # Send commands based on detected classes and scores
    if output_dict['detection_classes'][0] == 1 and output_dict['detection_scores'][0] > 0.70: 
        print('EYE DOWN')
        ser.write('2'.encode())
        sleep(10)
        ser.write('5'.encode())
    if output_dict['detection_classes'][0] == 2 and output_dict['detection_scores'][0] > 0.70:
        print('EYE LEFT')
        ser.write('4'.encode())
        sleep(10)
        ser.write('5'.encode())
    if output_dict['detection_classes'][0] == 3 and output_dict['detection_scores'][0] > 0.70:
        print('EYE RIGHT')
        ser.write('3'.encode())
        sleep(10)
        ser.write('5'.encode())
    if output_dict['detection_classes'][0] == 4 and output_dict['detection_scores'][0] > 0.70:
        print('EYE UP')
        ser.write('1'.encode())
        sleep(10)
        ser.write('5'.encode())

    return output_dict

global a2
a1 = 0
a2 = 0

url = 'http://192.0.0.4:8080/shot.jpg'

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
                imgPath = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
                image_np = cv2.imdecode(imgNp, -1)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                cv2.imwrite('capture.jpg', image_np)
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
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
except Exception as e:
    print(e)
#..............................................................................................................................!

import os
import pyttsx3
import json
import vosk
import pyaudio
import urllib.request
import numpy as np
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
import serial
from time import sleep
import cv2

# Initialize serial communication
ser = serial.Serial("COM3", baudrate='9600', timeout=0.5)

# Initialize TTS engine for feedback
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load the Vosk model
model_path = "E:/Programs/speech control/speechenv/vosk-model-small-en-in-0.4/vosk-model-small-en-in-0.4"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model = vosk.Model(model_path)
recognizer = vosk.KaldiRecognizer(model, 16000)

# Initialize PyAudio to capture microphone input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Check TensorFlow version
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Setup TensorFlow model
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
    with graph.as_default():
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        tensor_dict = {
            'num_detections': tf.get_default_graph().get_tensor_by_name('num_detections:0'),
            'detection_boxes': tf.get_default_graph().get_tensor_by_name('detection_boxes:0'),
            'detection_scores': tf.get_default_graph().get_tensor_by_name('detection_scores:0'),
            'detection_classes': tf.get_default_graph().get_tensor_by_name('detection_classes:0'),
        }

        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

def recognize_speech():
    print("Listening for voice command...")
    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_json = json.loads(result)
                command = result_json.get("text", "")
                if command:
                    return command.lower()
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        return None

def handle_voice_command(command):
    print(f"Handling voice command: {command}")  # Debugging line
    if "forward" in command:
        speak("Moving forward")
        ser.write('2'.encode())  # Send forward command to serial
    elif "backward" in command:
        speak("Moving backward")
        ser.write('5'.encode())  # Send backward command to serial
    elif "left" in command:
        speak("Turning left")
        ser.write('4'.encode())  # Send left command to serial
    elif "right" in command:
        speak("Turning right")
        ser.write('3'.encode())  # Send right command to serial
    elif "stop" in command:
        print("Braking")  # Debugging line
        speak("Stopping")
        ser.write('1'.encode())  # Send stop command to serial
    else:
        speak("Unknown voice command")

def handle_eye_input(eye_direction):
    print(f"Handling eye input: {eye_direction}")  # Debugging line
    if "forward" in eye_direction:
        speak("Moving forward")
        ser.write('2'.encode())  # Send forward command to serial
    elif "backward" in eye_direction:
        speak("Moving backward")
        ser.write('5'.encode())  # Send backward command to serial
    elif "left" in eye_direction:
        speak("Turning left")
        ser.write('4'.encode())  # Send left command to serial
    elif "right" in eye_direction:
        speak("Turning right")
        ser.write('3'.encode())  # Send right command to serial
    elif "stop" in eye_direction:
        print("Braking")  # Debugging line
        speak("Stopping")
        ser.write('1'.encode())  # Send stop command to serial
    else:
        speak("Unknown eye input")

# Main loop to continuously listen for commands and run object detection
if __name__ == "__main__":
    url = 'http://192.0.0.4:8080/shot.jpg'  # URL for camera feed

    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                # Process voice commands
                command = recognize_speech()
                if command:
                    print(f"Recognized voice command: {command}")
                    handle_voice_command(command)

                # Object detection logic
                imgPath = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
                image_np = cv2.imdecode(imgNp, -1)

                # Actual detection
                output_dict = run_inference_for_single_image(image_np, detection_graph)

                # Visualization of the results of a detection
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))

                # Eye input handling (replace this with your actual eye tracking logic)
                # For demonstration, let's assume you have a function to detect eye direction:
                eye_direction = "forward"  # This should come from your eye tracking logic
                handle_eye_input(eye_direction)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
