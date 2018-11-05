import cv2


SYSTEM_PATH = "D:/Python/DoAn/TrackingTraffic/Traffic-Tracking"

MODEL_NAME = SYSTEM_PATH + "/" + "trained"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/opt_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = SYSTEM_PATH + "/" + "data/graph.pbtxt"


cvNet = cv2.dnn.readNetFromTensorflow(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)

# cap = cv2.VideoCapture("D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/demo.mp4")

# ret = True
# while ret:
#     ret, img = cap.read()
#     rows = img.shape[0]
#     cols = img.shape[1]
#     cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
#     cvOut = cvNet.forward()
#     for detection in cvOut[0,0,:,:]:
#         score = float(detection[2])
#         if score > 0.3:
#             left = detection[3] * cols
#             top = detection[4] * rows
#             right = detection[5] * cols
#             bottom = detection[6] * rows
#             cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=1)
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) == 13:
#         break
# cap.release()
# cv2.destroyAllWindows()

