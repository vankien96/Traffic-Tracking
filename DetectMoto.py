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
import Upload as upload
import sort
from collections import OrderedDict

from collections import defaultdict

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import visualization_utils as vis_util

SYSTEM_PATH = "D:/Python/DoAn/TrackingTraffic/Traffic-Tracking"
MODEL_NAME = SYSTEM_PATH + "/" + "trained"
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
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

sess = tf.Session(graph=detection_graph)
# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        

def detect_moto(image):
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            imageclone = image.copy()
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
            return box_to_color_map

