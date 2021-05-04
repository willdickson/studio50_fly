import cv2
import numpy as np
from studio50_fly.utility import get_monitor_dict
from studio50_fly.blob_finder import BlobFinder

blob_finder = BlobFinder(threshold=30, min_area=50)

frame_width = 1280 
frame_height = 1024 

monitor_dict = get_monitor_dict()
user_monitor = monitor_dict['HDMI-0']
proj_monitor = monitor_dict['DP-1']

cv2.namedWindow('projector')
cv2.moveWindow('projector', proj_monitor.x, proj_monitor.y)
proj_image = 255*np.ones((proj_monitor.height, proj_monitor.width, 3) ,dtype=np.uint8)
cv2.imshow('projector', proj_image)

cap = cv2.VideoCapture('/dev/video0')
if not cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')):
    raise RuntimeError('unable to set fourcc to mjpg')
if not cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width):
    raise RuntimeError('unable to set frame width')
if not cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height):
    raise RuntimeError('unable to set frame height')
if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
    raise RuntimeError('unable to set auto exposure')
if not cap.set(cv2.CAP_PROP_EXPOSURE, 200):
    raise RuntimeError('unable to set auto exposure')

#cv2.namedWindow('camera')
#cv2.moveWindow('camera', user_monitor.width - frame_width, 0)

cv2.namedWindow('background')
cv2.moveWindow('background', user_monitor.width - frame_width, 0)


have_bg = False
done = False
cnt = 0

while not done:
    ret, image = cap.read()
    if ret:
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if have_bg:
            diff_image = cv2.absdiff(gray_image,bg_image) 
            blob_list, blob_image, thresh_image = blob_finder.find(diff_image)
            print(blob_list)
            #cv2.imshow('camera', image)
            cv2.imshow('blob_image', blob_image)
        else:
            gray_image = cv2.medianBlur(gray_image, 5)
            if cnt < 3:
                bg_image = np.zeros(gray_image.shape, dtype=np.uint8)
            else:
                pass
                bg_image = np.maximum(bg_image, gray_image)
            cv2.imshow('background', bg_image)

        cnt += 1

    key = cv2.waitKey(1) & 0xff
    if key == ord('g'):
        have_bg = True
        cv2.destroyWindow('background')
        cv2.namedWindow('blob_image')
        cv2.moveWindow('blob_image', user_monitor.width - frame_width, 0)
    if key == ord('q'): 
        done = True

cv2.destroyAllWindows()
cap.release()


