import cv2

image = cv2.imread("D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/image.JPG")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
ret, im_th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(im_th, 90, 200)
_, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(con) for con in contours]

for rect in rects:
    if rect[2] > 5*rect[3] and rect[3] > 30:
        cv2.rectangle(gray, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
        print(rect)
cv2.imshow("image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()