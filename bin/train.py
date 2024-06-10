import cv2 as cv
import os
import numpy as np
from PIL import Image


def train():
    path = "user"
    paths = [os.path.join(path, "shivesh", f) for f in os.listdir("user/shivesh")]


    faces = []
    ids = []

    for x, dir in enumerate(os.listdir("user")):
        for i, p in enumerate(paths):

            if "info" not in p:
                image = np.array(Image.open(p).convert('L'))
                id = x
                print(id)
                faces.append(image)
                ids.append(id)
            
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    recognizer.write('train_output/train.yml')