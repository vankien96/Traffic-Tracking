from firebase import firebase
from google.cloud import storage
import os
import datetime
import cv2
import pyrebase
import dropbox

config = {
    "apiKey": "AIzaSyCbr9x65lPqcwhL7tAvoF1Yec14zKBl_uQ",
    "authDomain": "haivan-211807.firebaseapp.com",
    "databaseURL": "https://haivan-211807.firebaseio.com",
    "projectId": "haivan-211807",
    "storageBucket": "haivan-211807.appspot.com",
    "messagingSenderId": "318452646500",
    "serviceAccount": "D:/Python/DoAn/TrackingTraffic/Traffic-Tracking/data/service.json"
}
camera_id = "DNG33"
database_path = "cameras/violation/"

firebase = pyrebase.initialize_app(config)
# Create a dropbox object using an API v2 key
drop = dropbox.Dropbox("glCCMkTizEAAAAAAAAAAC6exh6BP8bwA5cE7ahuzmv_jNZeOgVuqMmDs9IoeEDZt")


def upload_image(image, date_time, violation_count):
    image_path = "temp.jpg"
    image_name = str(datetime.datetime.now()) + ".jpg"
    cv2.imwrite(image_path, image)
    url = upload_to_dropbox(image_path, image_name)
    database = firebase.database()
    autoID = database.child(database_path).child(camera_id).generate_key()
    data = {
        "image_link": url,
        "timestamp": date_time,
        "number_of_moto_violation": violation_count
    }
    database.child(autoID).set(data)

def upload_to_dropbox(filepath, upload_filename):
	target = "/videos/DNG33/violation/"              # the target folder
	targetfile = target + upload_filename   # the target path and file name
	with open(filepath, "rb") as f:
		meta = drop.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))
	# create a shared link
	link = drop.sharing_create_shared_link(targetfile)
	# url which can be shared
	url = link.url
	return url