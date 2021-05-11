from __future__ import print_function
import cv2
import numpy as np

class BlobFinder(object):

    def __init__(self, threshold=200, filter_by_area=True, min_area=100, max_area=None, mask=None, invert=False):
        self.threshold = threshold
        self.filter_by_area = filter_by_area 
        self.min_area = min_area 
        self.max_area = max_area 
        self.invert = invert
        self.mask = mask

    def find(self,image):

        if self.invert:
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            threshold_type = cv2.THRESH_BINARY

        if self.threshold == 'otsu':
            threshold_type += cv2.THRESH_OTSU

        rval, thresh_image = cv2.threshold(image, self.threshold, 255, threshold_type)
        if self.mask is not None:
            thresh_image = np.bitwise_and(thresh_image, self.mask)

        contour_list, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_list = []
        blob_contours = []

        for contour in contour_list:

            blob_ok = True
            # Get area and apply area filter  
            area = cv2.contourArea(contour)
            if self.filter_by_area:
                if area <= 0:
                    blob_ok = False
                if self.min_area is not None:
                    if area < self.min_area:
                        blob_ok = False
                if self.max_area is not None:
                    if area > self.max_area:
                        blob_ok = False

            # Get centroid
            moments = cv2.moments(contour)
            if moments['m00'] > 0 and blob_ok:
                centroid_x = int(np.round(moments['m10']/moments['m00']))
                centroid_y = int(np.round(moments['m01']/moments['m00']))
            else:
                blob_ok = False
                centroid_x = 0
                centroid_y = 0

            # Get bounding rectangle
            if blob_ok:
                bounding_box = cv2.boundingRect(contour)
            else:
                bounding_box = (0,0,0,0)

            # Create blob dictionary
            blob = {
                    'moments'      : moments,
                    'centroid_x'   : centroid_x,
                    'centroid_y'   : centroid_y,
                    'bounding_box' : bounding_box,
                    'area'         : area,
                    'contour'      : contour,
                    } 

            # If blob is OK add to list of blobs
            if blob_ok: 
                blob_list.append(blob)
                blob_contours.append(contour)

        # Draw blob on image
        blob_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(blob_image,blob_contours,-1,(0,0,255),1)

        return blob_list, blob_image, thresh_image






