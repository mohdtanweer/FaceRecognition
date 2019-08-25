# Introduction 
This project is do find similarity between two images. It uses the deep learning features of OpenCV and is developed by following the guide of PyImageSearch.
It uses the pre-trained Caffe deep learning model provided by OpenCV to detect faces.
The face recognition model OpenCV uses to compute the 128-d face embeddings comes from the [OpenFace project](https://cmusatyalab.github.io/openface/)

# Getting Started

## Software Dependencies
1. Python 3.6
2. Pillow 6.0.0
3. opencv 4.1.0
5. opencv-contrib-python 4.1.0

### User Guide
Run the below command to get the cosine similarity between id and selfie images:

`python face_matching.py -id <path to id image> -s <path to selfie image> -d <path to face detection model> -m <path to embedding model> -o <path to output directory>`
