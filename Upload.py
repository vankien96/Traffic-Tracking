from firebase import firebase
from google.cloud import storage
import os
import datetime
import cv2

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/data/service.json"
firebase = firebase.FirebaseApplication("https://testproject-4e77f.firebaseio.com/", authentication=None)

def upload_image(image):
    image_name = "temp.jpg"
    cv2.imwrite(image_name, image)
    client = storage.Client()
    bucket = client.get_bucket("testproject-4e77f.appspot.com")
    imageBlob = bucket.blob("image/" + str(datetime.datetime.now()) + ".jpg")
    imageBlob.upload_from_filename(image_name)
    imageBlob.make_public()
    firebase.post('/data', imageBlob.public_url)