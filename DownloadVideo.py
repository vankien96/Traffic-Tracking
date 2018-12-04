import requests
import json
import base64
import wget
import numpy as np
import cv2
from lxml import html
import requests
import datetime

CAMERA_ID = "DNG33"
DATAPOINT = 'http://2co2.vp9.tv/chn/'+ CAMERA_ID +'/'
PATH_TO_CACHE_DOWNLOAD_VIDEOS  = 'cache/'

last_time = datetime.datetime.now()
time_distance = 0
video_paths = []

def crawl_and_handle_video():
    global last_time, time_distance
    page = requests.get(DATAPOINT)
    if (page.ok):
        try:
            tree = html.fromstring(page.content)
            videos = tree.xpath("//a[contains(., '.ts')]/text()")
            dates = tree.xpath("//td[contains(., ':')]/text()")
            date = dates[-1]
            filename = videos[-1]
            now = datetime.datetime.now()
            distance_time_in_second = (now - last_time).total_seconds()
            if distance_time_in_second > time_distance:
                last_time = now
                video_url = DATAPOINT + filename
                date_time_now = datetime.datetime.now()
                filename = str(date_time_now) + ".ts"
                filename = filename.replace("-","_").replace(":","@").replace(" ","S")
                    
                video_path = PATH_TO_CACHE_DOWNLOAD_VIDEOS + filename
                wget.download(video_url, video_path)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                time_distance = frameCount/fps
                cap.release()
                video_paths.append(video_path)
        except:
            print("error")

def begin_download():
    while True:
        crawl_and_handle_video()
