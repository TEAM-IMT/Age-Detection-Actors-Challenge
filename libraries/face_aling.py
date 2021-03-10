# import the necessary packages
from Alignment import FaceAligner
from help import rect_to_bb
import argparse
import imutils
import dlib
import cv2

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread("40.jpg")
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Input", image)

(x, y, w, h) = (0 ,0 ,image.shape[1] ,image.shape[0] )
rect = dlib.rectangle(left=x, top=y, right=w, bottom=h)
faceOrig=imutils.resize(image[y:y + h, x:x + w], width=256)
faceAligned = fa.align(image, gray, rect)
# display the output images
cv2.imshow("Original", faceOrig)
cv2.imshow("Aligned", faceAligned)
cv2.waitKey(0)
