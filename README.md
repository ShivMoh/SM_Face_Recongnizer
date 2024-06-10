A simple python based application made to experiment and try out different face recongnition algorithms and techniqes. Currently implemented are:

1. Opencv's haarclassifier for face detection
2. Opencv's LBPHFaceRecognizer for face recongition
3. Facenet pytorch mtcnn for face recognition

In the future, will likely look into and implement more methods for face detection.

## Steps to Run

```
cd folder_name
python main.py
```

1. Type y if new user to collect data for training and wait until data collection is complete
2. Select your recongnizer of choice and wait for opencv window to show up
3. Press d to exit the application


## Notable requirments

opencv
pytorch
facenet-pytorch 
numpy
pillow

## Sources

Some sources used are listed below:

Facenet pytorch: https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
Mjrovia: https://github.com/Mjrovai/OpenCV-Face-Recognition/tree/master/FacialRecognition

