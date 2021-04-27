import cv2
import time
import numpy as np
from studio50_fly.utility import get_monitor_dict
from studio50_fly.blob_finder import BlobFinder

def get_max_area_blob(blob_list):
    if len(blob_list) == 0:
        max_blob = None
    else:
        area_array= np.array([b['area'] for b in blob_list])
        max_ind = np.argmax(area_array)
        max_blob = blob_list[max_ind]
    return max_blob


frame_width = 1280 
frame_height = 1024 

monitor_dict = get_monitor_dict()
user_monitor = monitor_dict['HDMI-0']
proj_monitor = monitor_dict['DP-1']

cv2.namedWindow('projector')
cv2.moveWindow('projector', proj_monitor.x, proj_monitor.y)


cap = cv2.VideoCapture('/dev/video0')
if not cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')):
    raise RuntimeError('unable to set fourcc to mjpg')
if not cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width):
    raise RuntimeError('unable to set frame width')
if not cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height):
    raise RuntimeError('unable to set frame height')
if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3):
    raise RuntimeError('unable to set auto exposure')
#if not cap.set(cv2.CAP_PROP_EXPOSURE, 400):
#    raise RuntimeError('unable to set exposure')

cv2.namedWindow('camera')
cv2.moveWindow('camera', user_monitor.width - frame_width, 0)
cv2.namedWindow('blob_image')
cv2.moveWindow('blob_image', user_monitor.width - frame_width, 0)

done = False

blob_finder = BlobFinder(threshold=40,min_area=50)


proj_image = np.zeros((proj_monitor.height, proj_monitor.width, 3) ,dtype=np.uint8)
cv2.imshow('projector', proj_image)
cv2.waitKey(1)
time.sleep(0.1)

cnt = -10 

min_x, max_x = -200, 200
min_y, max_y = -200, 200
step = 20
ind_list = [(x,y) for x in range(min_x,max_x,step) for y in range(min_y,max_y,step)]

projector_pts = []
camera_pts = []


while not done:
    ret, image = cap.read()

    if ret:
        if cnt > 0: 
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blob_list, blob_image, thresh_image = blob_finder.find(gray_image)
            blob = get_max_area_blob(blob_list)
            cv2.imshow('camera', image)
            cv2.imshow('blob_image', blob_image)
            if blob is not None: 
                blob_x = blob['centroid_x']
                blob_y = blob['centroid_y']
                camera_pts.append((blob_x, blob_y))
                projector_pts.append((dot_x, dot_y))
        if cnt >= 0:
            proj_image = np.zeros((proj_monitor.height, proj_monitor.width, 3) ,dtype=np.uint8)
            dot_x = proj_monitor.width//2 + ind_list[cnt][0] 
            dot_y = proj_monitor.height//2 + ind_list[cnt][1]
            proj_image = cv2.circle(proj_image, (dot_x,dot_y), 5, (255,255,255), cv2.FILLED, cv2.LINE_8,0)

    cv2.imshow('projector', proj_image)
    cnt += 1

    if cnt >= len(ind_list):
        done = True

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): 
        done = True

cv2.destroyAllWindows()
cap.release()

print(len(camera_pts))


