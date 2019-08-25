import os, sys
import cv2
import time
import scipy
import numpy as np

from scipy import spatial

from face_extraction import FaceExtraction
from face_embedding import FaceEmbedding

import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-id", "--id-image", required=True,
                help="path to identification card image")
ap.add_argument("-s", "--selfie-image", required=True,
                help="path to selfie image")
ap.add_argument("-d", "--detector", required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output-dir", required=True,
                help="path to save the output image")

args = vars(ap.parse_args())

def main():
    print("In main")
    idImage = cv2.imread(args['id_image'])
    selfieImage = cv2.imread(args['selfie_image'])
    modelPath = args['detector']
    embeddingModel = args['embedding_model']
    out_dir = args['output_dir']

    # Get feature vector for ID image
    detect = FaceExtraction(idImage, modelPath)
    faces = detect.detect_face()
    if len(faces) > 1:
        print("More than 1 faces detected in the ID image\nPlease provide another ID!!!")
    else:
        cv2.imshow("Face", faces[0])
        cv2.imwrite(out_dir + 'A001.png', faces[0])
        faceEmbeddingVec = FaceEmbedding(faces[0], embeddingModel)
        embeddingVectorId = faceEmbeddingVec.get_face_embedding()

    # Get feature vector for Selfie image
    detect = FaceExtraction(selfieImage, modelPath)
    faces = detect.detect_face()
    if len(faces) > 1:
        print("More than 1 faces detected in the Selfie\nPlease provide another Selfie!!!")
    else:
        cv2.imshow("Face", faces[0])
        cv2.imwrite(out_dir + 'B001.png', faces[0])
        faceEmbeddingVec = FaceEmbedding(faces[0], embeddingModel)
        embeddingVectorSelfie = faceEmbeddingVec.get_face_embedding()

    # Get cosine distance between id and selfie images
    similarity_dist = spatial.distance.cosine(embeddingVectorId, embeddingVectorSelfie)
    print("Similarity between the images: {}".format(similarity_dist))


if __name__ == '__main__':
    main()
