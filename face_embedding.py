# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

class FaceEmbedding():
    def __init__(self, img, modelPath):
        self.img = img
        self.modelPath = modelPath


    def get_face_embedding(self):
        embedder = cv2.dnn.readNetFromTorch(self.modelPath)

        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv2.dnn.blobFromImage(self.img, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        return vec
