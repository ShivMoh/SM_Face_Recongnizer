import cv2 as cv
import numpy as np
import os
from datetime import datetime
from PIL import Image


class Data:
    opencv_path = "opencv_haarcascade"
    path = os.path.join(opencv_path, "front.xml")

    video = None
    haar_cascade = None
    cropped_face = (0, 0)
    user_folder = "users"

    prompt = "Hello, new user! How are you? Could I kindly get your name please"
  
    count = 0

    user_path = ""

    init = True

    def __init__(self, iterations = 200):
        self.video = cv.VideoCapture(0)
        self.haar_cascade = cv.CascadeClassifier(self.path)
        self.user = input(self.prompt)
        self.collect_data(iterations)
        self.update_info_text()

    def collect_data(self, iterations):
        print("Collecting data")
        for x in range(iterations):
            ret, frame = self.video.read()
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces_rect = self.haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=3)
            user_path = os.path.join(self.user_folder, self.user)

            if not os.path.exists(user_path):
                os.makedirs(user_path)
                if (not os.path.exists(os.path.join(user_path, "info.txt"))):
                    f = open(os.path.join(user_path, "info.txt"), "w")
                    f.write(f"image_count:{self.count}\n")
                    f.write(f"date_created:{datetime.now()} \n")
                    f.write(f"last_updated: {datetime.now()}\n")
                    print("File created")
                    self.init = False
                    f.close()

            if self.init:
                l = []
                with open(os.path.join(user_path, "info.txt"), "r") as file:
                    l = file.readlines()
                self.count = int(l[0].split(":")[1])
                self.init = False
                file.close()
                
            if (len(faces_rect) != 0):    
                for (x,y,w,h) in faces_rect:
                    cropped_face = gray_frame[y:y+h, x:x+w]
                    cv.rectangle(frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
                
                cv.imwrite(os.path.join(user_path, f"{self.user}-{self.count}.png"), cropped_face)    
                self.count+=1

            
            cv.imshow("Cropped face", cropped_face)
            cv.imshow("Frame", frame)

            if cv.waitKey(20) & 0xFF==ord('d'):
                break

        self.video.release()
        cv.destroyAllWindows()

    def update_info_text(self):
        lines = []

        with open(f"{self.user_folder}/{self.user}/info.txt", "r") as file:
            lines = file.readlines()
            
        for i, line in enumerate(lines):
            if "image_count" in line:
                lines[i]=f"image_count:{self.count}\n"
            
            if "last_updated" in line:
                lines[i]=f"last_updated:{datetime.now()}\n"

        file = open(f"{self.user_folder}/{self.user}/info.txt", "w")

        for line in lines:
            file.write(line)

        file.close()

 