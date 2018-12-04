import numpy as np
import cv2
import DetectLight as lightDec
import lineInfo as line
import CentroidTracker
import Draw as draw
import Upload as upload
import sort
import DownloadVideo as download
import DetectMoto as detectMoto
from collections import OrderedDict
from threading import Thread
import os
from os import listdir
from os.path import isfile, join

def detect_violate(video_path):
  cap = cv2.VideoCapture(video_path)
  date_time = video_path.replace("cache/", "").replace(".ts", "").replace("_", "-").replace("S", " ").replace("@", ":")
  date_time = date_time[:-10]
  tracker = sort.Sort()
  location_before_red_light = OrderedDict()
  violate_moto = []
  ret = True
  while ret:
    ret, image = cap.read()
    if ret == False:
      break
    box_of_objects = detectMoto.detect_moto(image)
    

    is_red = lightDec.is_red_light(image)
    dets = []

    for box, score in box_of_objects.items():
      ymin, xmin, ymax, xmax = box
      im_height, im_width = image.shape[:2]
      (start_x, end_x, start_y, end_y) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
      dets.append(np.array([start_x, start_y, end_x, end_y, score]))
          
    dataTrack = tracker.update(np.array(dets))

    have_new_moto_violate = False
    count_moto = 0
    for data in dataTrack:
      (start_x, start_y, end_x, end_y) = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
      trackingID = int(data[4])
      draw.put_objectID_into_object(image, (start_x, start_y), trackingID)
      cv2.rectangle(image, (start_x, start_y), (end_x, end_y), [0, 255, 0], 1)
      if is_red:
        if trackingID not in location_before_red_light.keys():
          location_before_red_light[trackingID] = (start_x, start_y, end_x, end_y)
        initLocation = location_before_red_light[trackingID]

        # check if location when light turn red and current 
        # if moto cross the line
        if end_y < line.line_center_y and initLocation[3] > line.line_center_y and end_y < initLocation[3]:
          cv2.rectangle(image, (start_x, start_y), (end_x, end_y), [0, 0, 255], 1)
          count_moto += 1
          if trackingID not in violate_moto:
            violate_moto.append(trackingID)
            have_new_moto_violate = True
            
    if have_new_moto_violate:
      upload_thread = Thread(target=upload.upload_image, args=(image, date_time, count_moto))
      upload_thread.start()

    # out.write(image)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == 13:
      break
  cap.release()
  cv2.destroyAllWindows()

def process_video():
  i = 0 
  while True:
    if len(download.video_paths) > i:
      detect_violate(download.video_paths[i])
      os.remove(download.video_paths[i])
      i += 1

# download_thread = Thread(target=download.begin_download, args=())
# download_thread.start()
# process_thread = Thread(target=process_video, args=())
# process_thread.start()

filenames = [f for f in listdir("cache") if isfile(join("cache", f))]
for filename in filenames:
  filepath = "cache/"+filename
  detect_violate(filepath)
  # os.remove(filepath)


