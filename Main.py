from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import DetectLight as lightDec
import lineInfo as line
import CentroidTracker
import Draw as draw
import sort

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

SYSTEM_PATH = "D:/models/research/object_detection"

MODEL_NAME = SYSTEM_PATH + "/" + "trained"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = SYSTEM_PATH + "/" + "data/object-detection.pbtxt"

NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    graph_def.ParseFromString(serialized_graph)
  tf.import_graph_def(graph_def, name='')

label_map_a = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map_a, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.shape[:2]
  return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

### line not detect y = 0.8*x - 401.5
a = 0.8
b = -401.5
cap = cv2.VideoCapture("D:/Python/DoAn/TrackingTraffic/demo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("D:/Python/DoAn/TrackingTraffic/output2.avi", fourcc, 20.0, (1280, 720) )
ret = True
ct = CentroidTracker.CentroidTracker()
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    ret = True
    while ret:
        ret, image_np = cap.read()
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        imageclone = image_np.copy()
        image, box_to_color_map, box_of_string = vis_util.visualize_boxes_and_labels_on_image_array(
          imageclone,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1, 
          skip_labels=False,
          skip_scores=True)

        
        for box, nameObject in box_of_string.items():
          if nameObject[0] == 'car':
            del box_to_color_map[box]

        is_red = lightDec.is_red_light(image_np)
        rects = []
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            im_height, im_width = image_np.shape[:2]
            (start_x, end_x, start_y, end_y) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
            rects.append((start_x, start_y, end_x, end_y))
            cv2.rectangle(image_np, (start_x, start_y), (end_x, end_y), [0, 255, 0], 1)
        
        ct.update(rects)
        for (objectID, location) in ct.currentLocation.items():
            rect = (location[0], location[1], location[2], location[3])
            if rect in rects:
                cv2.rectangle(image_np, (location[0], location[1]), (location[2], location[3]), [0, 255, 0], 1)
                draw.put_objectID_into_object(image_np, (location[0], location[1]), objectID)
            if is_red:
                centroid = ct.objects[objectID]
                if objectID not in ct.initLocation.keys():
                  ct.initLocation[objectID] = centroid
                initCentroid = ct.initLocation[objectID]
                if initCentroid[1] > line.line_center_y and centroid[1] < line.line_center_y:
                    cv2.rectangle(image_np, (location[0], location[1]), (location[2], location[3]), [0, 0, 255], 1)
          

        out.write(image_np)
        cv2.imshow("image", image_np)
        if cv2.waitKey(1) == 13:
            break
cap.release()
cv2.destroyAllWindows()

