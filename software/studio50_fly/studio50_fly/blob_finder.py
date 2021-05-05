from __future__ import print_function
import cv2
import numpy as np

class BlobFinder(object):

    def __init__(self, threshold=200, filter_by_area=True, min_area=100, max_area=None):
        self.threshold = threshold
        self.filter_by_area = filter_by_area 
        self.min_area = min_area 
        self.max_area = max_area 

    def find(self,image):
        if self.threshold == 'otsu':
            rval, thresh_image = cv2.threshold(image, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            rval, thresh_image = cv2.threshold(image, self.threshold,255,cv2.THRESH_BINARY)
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
                bound_rect = cv2.boundingRect(contour)
                min_x = bound_rect[0]
                min_y = bound_rect[1]
                max_x = bound_rect[0] + bound_rect[2] 
                max_y = bound_rect[1] + bound_rect[3] 
            else:
                min_x = 0.0 
                min_y = 0.0
                max_x = 0.0
                max_y = 0.0

            # Create blob dictionary
            blob = {
                    'moments'    : moments,
                    'centroid_x' : centroid_x,
                    'centroid_y' : centroid_y,
                    'min_x'      : min_x,
                    'max_x'      : max_x,
                    'min_y'      : min_y,
                    'max_y'      : max_y,
                    'area'       : area,
                    } 

            # If blob is OK add to list of blobs
            if blob_ok: 
                blob_list.append(blob)
                blob_contours.append(contour)

        # Draw blob on image
        blob_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(blob_image,blob_contours,-1,(0,0,255),1)

        return blob_list, blob_image, thresh_image






