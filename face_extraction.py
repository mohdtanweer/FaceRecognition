# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

class FaceExtraction():
    def __init__(self, img, modelPath):
        self.img = img
        self.modelPath = modelPath
        self.threshold = 0.5
        self.dim = (96, 112) # Setting dimension to feed into Deep Learning Model for recognition

    def detect_face(self):
        protoPath = self.modelPath + "deploy.prototxt"
        model = self.modelPath + "res10_300x300_ssd_iter_140000.caffemodel"

        detector = cv2.dnn.readNetFromCaffe(protoPath, model)

        # Resize the image to width = 600 maintaining aspect ratio
        image = imutils.resize(self.img, width=600)
        h, w = image.shape[:2]
        # Creating Image Blob
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                          (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        # Loop over the detections
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                faces.append(face)

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
        return faces
