import cv2
import time
import numpy as np
from studio50_fly.utility import get_monitor_dict

def find_arena(image, threshold=100):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rval, threshold_image = cv2.threshold(gray_image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY)
    contour_list, dummy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area_array = np.array([cv2.contourArea(c) for c in contour_list])
    if area_array.size < 1:
        contour_data = {
                'area' : 0,
                'contour': None,  
                'centroid_x': 0,
                'centroid_y': 0,
                'contour_image': image,
                'bounding_box' : None,
                }
    else:
        ind_of_max = area_array.argmax()
        area = area_array[ind_of_max]
        contour = contour_list[ind_of_max]
        moments = cv2.moments(contour) 
        centroid_x = int(np.round(moments['m10']/moments['m00']))
        centroid_y = int(np.round(moments['m01']/moments['m00']))
        bounding_box = cv2.boundingRect(contour)

        contour_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image,[contour],-1,(0,0,255),2)
        cv2.circle(contour_image,(int(centroid_x),int(centroid_y)),10,(255,0,),2)
        bbx, bby, bbw, bbh = bounding_box
        cv2.rectangle(contour_image,(bbx,bby),(bbx+bbw,bby+bbh),(0,255,0),2)

        contour_data = {
                'area' : area,
                'contour': contour,  
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'contour_image': contour_image,
                'bounding_box' : bounding_box,
                }
    return contour_data


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

cv2.namedWindow('camera')
cv2.moveWindow('camera', user_monitor.width - frame_width, 0)

cv2.namedWindow('arena')
cv2.moveWindow('arena', user_monitor.width - frame_width, 0)

done = False

while not done:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('camera', frame)
        arena_data = find_arena(frame, threshold=50)
        cv2.imshow('arena', arena_data['contour_image'])
        print(f"{arena_data['centroid_x']}, {arena_data['centroid_y']}")

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): 
        done = True

cv2.destroyAllWindows()
cap.release()


