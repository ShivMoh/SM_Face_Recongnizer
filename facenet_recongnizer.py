from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
from PIL import Image
from recognizer import Recongnizer
import cv2 as cv
import time

"""
    Code adopted from https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
"""

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def collate_fn(x):
    return x[0]

class FaceNet(Recongnizer):
    resnet = None
    mtcnn = None
    dataset = []
    loader = []
    embeddings = []    
    
    def __init__(self)->None:
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device
        )
        self.batch_size = 8
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.train()

    def train(self):
        self.dataset = datasets.ImageFolder('./users')
        self.dataset.idx_to_class = {i:c for c, i in self.dataset.class_to_idx.items()}
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=workers)

        self.calculate_embeddings()

    def calculate_embeddings(self):
        print("Calculating embeddings")
        aligned = []
        names = []
        for x, y in self.loader:
            with torch.no_grad():
                x_aligned, prob = self.mtcnn(x, return_prob=True)
                if x_aligned is not None:
                    aligned.append(x_aligned)
                    names.append(self.dataset.idx_to_class[y])
        
        current_name = names[0]
        class_embeddings = []

        with torch.no_grad():
            for index, name in enumerate(names):
                if name != current_name:
                    current_name = name
                    self.embeddings.append(class_embeddings)      
                    class_embeddings = []   

                embedding = self.resnet((aligned[index]).unsqueeze(0).to(device)).detach().cpu()
                class_embeddings.append(embedding)
        
        self.embeddings.append(class_embeddings)
  
    def predict_class(self, image):

        input_image = Image.fromarray(image).convert('RGB')

        img_cropped = self.mtcnn(input_image)

        if img_cropped is None: 
            return ["No Image", 0]
        
        img_embedding = self.resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu()

        distances = []

        for class_embedding in self.embeddings:
            distance_s = [(img_embedding - e).norm().item() for e in class_embedding]
            distances.append(distance_s)
            
        min_dist = 100
        max_dist = 1

        min_idx = 0

        for index, distance_s in enumerate(distances):
            if min(distance_s) > 1:
                max_dist = min(distance_s)
                
            if min_dist > min(distance_s):
                min_dist = min(distance_s)
                min_idx = index
   

        confidence = (max_dist - min_dist) * 100

        return [self.dataset.idx_to_class[min_idx], confidence]

# start = time.time()
# facenet = FaceNet()
# image = cv.imread("./users/john/john-37.png")
# facenet.train()
# print(facenet.predict_class(image))

# end = time.time()

# print("Total time", end - start)
# facenet.train()

# for i in range(10):
#     print(facenet.predict_class())