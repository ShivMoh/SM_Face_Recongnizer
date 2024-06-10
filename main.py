

import cv2 as cv
import numpy as np
import os 
from data import Data
from recognizer import Recongnizer
from haarclassifier_recognizer import Haarclassifier
from facenet_recongnizer import FaceNet

class Predictor:

    recognizer = None
    names = None
    camera = None

    mingW = 0
    mingH = 0

    cascadePath = "opencv_haarcascade/front.xml"
    font = cv.FONT_HERSHEY_SIMPLEX
    user_id = 0
    face = np.zeros((10, 10))
    names = []

    recognizer = None

    def __init__(self, recognizer : Recongnizer) -> None:
        self.names = os.listdir('users')
        self.camera = cv.VideoCapture(0)
        self.camera.set(3, 640) 
        self.camera.set(4, 480) 
        self.min_w = 0.1*self.camera.get(3)
        self.min_h = 0.1*self.camera.get(4)
        self.faceCascade = cv.CascadeClassifier(self.cascadePath)
        self.recognizer = recognizer

    def run(self):
        while True:

            ret, frame = self.camera.read()

            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(self.min_w), int(self.min_h)),
            )

            for(x,y,w,h) in faces:

                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                self.face = frame[y:y+h, x:x+w]

                user, confidence = self.recognizer.predict_class(gray[y:y+h,x:x+w])
            
                cv.putText(frame, str(user) + " " + str(round(confidence)) + "%", (x+5,y+20), self.font, 0.8, (255,255,255), 1)
            
            cv.imshow('face',self.face) 
            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xff==ord('d'):
                break

        self.camera.release()
        cv.destroyAllWindows()

user = input("Are you a new user? y\\n")

if (user == "y"):
    Data()

recognizer = None

recognizer_choice = input("""
Please input the number for the recognizer you'd like to use:
1. Open CV HaarClassifer
2. MTCNN trained on facenet data
""")

if recognizer_choice.replace(" ", "") == "1":
    print("Using HaarClassifer")
    recognizer = Haarclassifier()
elif recognizer_choice.replace(" ", "") == "2":
    print("Using MTCNN")
    recognizer = FaceNet()
else:
    print("Defaulting to haar classifer")

predictor = Predictor(recognizer=recognizer)
predictor.run()
