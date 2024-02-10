import os
from ultralytics import YOLO
from PIL import Image
import cv2 as cv


directorys  = [] #list of directories
faces_IDS = [] #IDs of faces found
faces_Found = [] #list of faces, IDS, found for each image in directorys


def file_org (DIR):
    for dirpath, dirnames, filename in os.walk(DIR):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            directorys.append(dirpath)
            print (filename)


def face_detect (DIR):
    model = YOLO("runs/detect/train3/weights/last.pt")
    for filename in os.listdir(DIR):
        f = os.path.join(DIR, filename)
        #run detection and save predictions
        img = Image.open(f)
        results = model.predict(source=img, save=True)  # save plotted images
        print(str(results.path))
        faces_Found.append(results[0].boxes)
        faces_IDS.append(filename)
        print(faces_Found)
        print(faces_IDS)
        print(len(faces_Found))
        print(len(faces_IDS))
        DIR = os.path.join(results[0].save_dir,(os.listdir(results[0].save_dir))[0])
        print(DIR)