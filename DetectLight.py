import numpy as np
import cv2

#####################################################
# Detect light 
#####################################################
red_box_LT = (981, 214)
red_box_RB = (996, 228)

green_box_LT = (976, 246)
green_box_RB = (988, 259)

def av_color_intensity(img,x1,y1,x2,y2,col):
    col_intensity = 0
    for x in range(x1,x2+1):
        for y in range(y1,y2+1):
            col_intensity += img[y, x][col] - np.mean(img[y, x])
    # return average
    return col_intensity/( (x2-x1 + 1)*(y2-y1 + 1))

def is_red_light(image):
    global red_box_LT, green_box_LT
    red_box_intensity = av_color_intensity(image, red_box_LT[0], red_box_LT[1], red_box_RB[0], red_box_RB[1], 2)
    green_box_intensity = av_color_intensity(image, green_box_LT[0], green_box_LT[1], green_box_RB[0], green_box_RB[1], 1)
        # if there is at least 33% more red or green then choose that light!
        # update search area and add box
    if red_box_intensity > 1.33*green_box_intensity:
        draw_box_UL = (red_box_LT[0], red_box_LT[1])
        draw_box_LR = (red_box_RB[0] + 10, red_box_RB[1] + 10)
        cv2.rectangle(image, draw_box_UL, draw_box_LR, (0,0,255), 1)
        return True
    if green_box_intensity > 1.33*red_box_intensity:
        draw_box_UL = (green_box_LT[0], green_box_LT[1])
        draw_box_LR = (green_box_RB[0] + 10, green_box_RB[1] + 10)
        cv2.rectangle(image, draw_box_UL, draw_box_LR, (0,255,0), 1)
        return False
    return False
#####################################################
# Detect light 
#####################################################