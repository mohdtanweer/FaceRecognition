import os, sys
import cv2
import time
import numpy as np

from face_extraction import FaceExtraction
from face_embedding import FaceEmbedding

class TestFaceExtraction():
    def __init__(self, img, modelPath, embeddingModel):
        self.img = img
        self.modelPath = modelPath
        self.embeddingModel = embeddingModel
        self.extract_main()

    def extract_main(self):
        image = cv2.imread(self.img)

        detect = FaceExtraction(image, self.modelPath)
        faces = detect.detect_face()
        for face in faces:
            print("Shape of face detected: {}".format(face.shape))
            cv2.imshow("Face", face)
            # cv2.waitKey(0)
        print("Embedding Model: {}".format(self.embeddingModel))
        faceEmbeddingVec = FaceEmbedding(faces[0], self.embeddingModel)
        embeddingVector = faceEmbeddingVec.get_face_embedding()
        print(embeddingVector)


def main():
    startTime = time.time()
    img = "/Users/mohammadtanweer/Documents/Personal/Code/data/tanweer/A001.jpg"
    # img = "/Users/mohammadtanweer/Downloads/IMG-2189.JPG"
    modelPath = "/Users/mohammadtanweer/Documents/Personal/Projects/FaceRecognition/face_detection_model/"
    embeddingModel = "/Users/mohammadtanweer/Documents/Personal/Projects/FaceRecognition/face_embedding_model/openface_nn4.small2.v1.t7"
    TestFaceExtraction(img, modelPath, embeddingModel)
    print("Time taken to detect face: {}".format(time.time() - startTime))

if __name__ == '__main__':
    main()