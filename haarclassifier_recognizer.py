import cv2 as cv
from recognizer import Recongnizer
import os
from PIL import Image
import numpy as np

''''
Original code developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  : https://github.com/Mjrovai/OpenCV-Face-Recognition/tree/master/FacialRecognition
Adopted and altered by https://github.com/ShivMoh
'''
class Haarclassifier(Recongnizer):
    user_folder = "users"

    def __init__(self) -> None:
        super().__init__()
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        self.names = os.listdir('users')
        self.train()
        # self.recognizer.read('train/train.yml')

    def predict_class(self, image):
       
        user_id, confidence = self.recognizer.predict(image)

        # https://answers.opencv.org/question/226714/confidence-value-in-lbph/
        # https://answers.opencv.org/question/104724/confidence-in-face-recognition/

        confidence = 100*(1-(confidence/300))
        user_id = self.names[user_id]
        return [user_id, confidence]

    def train(self):
        print("Training")
        faces = []
        ids = []

        for x, dir in enumerate(os.listdir("users")):
         
            sub_dir = os.listdir(os.path.join("users", dir))
            for i, p in enumerate(sub_dir):
                if "info" not in p:
                    image = np.array(Image.open(os.path.join("users", dir, p)).convert('L'))
                    id = x
                    faces.append(image)
                    ids.append(id)
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))

        if not os.path.exists("/train"):
            os.mkdir("train")
            f = open("train/train.yml", "w")

        recognizer.write('train/train.yml')
        self.recognizer.read('train/train.yml')

