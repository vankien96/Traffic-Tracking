import numpy as np
import cv2
import DetectLight as lightDec
import lineInfo as line
import CentroidTracker
import Draw as draw
import Upload as upload
import sort
import DetectMoto as detectMoto
from collections import OrderedDict
from threading import Thread

cap = cv2.VideoCapture("D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/demo.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/output4.avi", fourcc, 13.0, (1280, 720) )
tracker = sort.Sort()

location_before_red_light = OrderedDict()
violate_moto = []
ret = True
while ret:
  ret, image = cap.read()
  if ret == False:
    break
  box_to_color_map = detectMoto.detect_moto(image)
  

  is_red = lightDec.is_red_light(image)
  dets = []

  for box, score in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    im_height, im_width = image.shape[:2]
    (start_x, end_x, start_y, end_y) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
    dets.append(np.array([start_x, start_y, end_x, end_y, score]))
        
  dataTrack = tracker.update(np.array(dets))

  for data in dataTrack:
    (start_x, start_y, end_x, end_y) = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
    trackingID = int(data[4])
    draw.put_objectID_into_object(image, (start_x, start_y), trackingID)
    if is_red:
      if trackingID not in location_before_red_light.keys():
        location_before_red_light[trackingID] = (start_x, start_y, end_x, end_y)
      initLocation = location_before_red_light[trackingID]

            # check if location when light turn red and current 
            # if moto cross the line
      if (end_y - 50) < line.line_center_y and initLocation[3] > line.line_center_y:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), [0, 0, 255], 1)
        if trackingID not in violate_moto:
          violate_moto.append(trackingID)
          upload_thread = Thread(target=upload.upload_image, args=(image,))
          upload_thread.start()

        draw.put_number_moto_violate(image, len(violate_moto))
        
  draw.put_number_moto_violate(image, len(violate_moto))

  out.write(image)
  cv2.imshow("image", image)
  if cv2.waitKey(1) == 13:
    break
cap.release()
cv2.destroyAllWindows()

