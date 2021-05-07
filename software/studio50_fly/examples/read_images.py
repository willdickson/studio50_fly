import sys
import h5py
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = h5py.File(filename,'r')

image_array = data['image']

for image in image_array:
    cv2.imshow('images', image)
    cv2.waitKey(10)



