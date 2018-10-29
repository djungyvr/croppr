import cv2
import numpy as np
import argparse
import imutils
import os

from imutils.object_detection import non_max_suppression

parser = argparse.ArgumentParser(description='Cropping photos')
parser.add_argument('-i', metavar='INPUT', type=str, help='Input directory')
parser.add_argument('-o', metavar='OUTPUT', type=str, help='Output image')
parser.add_argument('-s', metavar='SUFFIX', type=str, help='Suffix match')
parser.add_argument('-d', metavar='DIMENSION', type=str, help='Output dimension square')

args = parser.parse_args()

input_dir = args.i
output_dir = args.o
suffix_match = args.s
dim = float(args.d)

os.mkdir(output_dir)

files = [f for f in os.listdir(input_dir) if f.endswith(suffix_match)]

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for f in files:
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = cv2.imread(os.path.join(input_dir, f))
    image = imutils.resize(image, width=352, height=240)
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.7)

    # draw the final bounding boxes
    if len(pick) == 0:
        continue
    xA, yA, xB, yB = pick[0]

    crop_image = image[yA:yB, xA:xB]

    fy = dim/crop_image.shape[0]
    fx = dim/crop_image.shape[1]
    resize_image = cv2.resize(crop_image, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    output_file_path = os.path.join(output_dir, f)
    cv2.imwrite(output_file_path, resize_image)
