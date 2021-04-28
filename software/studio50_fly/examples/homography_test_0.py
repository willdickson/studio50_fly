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


blob_finder = BlobFinder(threshold=40,min_area=50)

min_x, max_x = -250, 250
min_y, max_y = -250, 250
step = 50 
ind_list = [(x,y) for x in range(min_x,max_x,step) for y in range(min_y,max_y,step)]

projector_pts = []
camera_pts = []


collection_done = False
pt_count = 0

while not collection_done:

    proj_image = np.zeros((proj_monitor.height, proj_monitor.width, 3) ,dtype=np.uint8)
    dot_x = proj_monitor.width//2 + ind_list[pt_count][0] 
    dot_y = proj_monitor.height//2 + ind_list[pt_count][1]
    proj_image = cv2.circle(proj_image, (dot_x,dot_y), 10, (255,255,255), cv2.FILLED, cv2.LINE_8,0)
    cv2.imshow('projector', proj_image)
    key = cv2.waitKey(100) & 0xff
    if key == ord('q'): 
        collection_done = True


    try_count = 0
    max_try_count = 20

    blob_capture_count = 0
    required_blob_capture_count = 10 

    blob = None
    blob_finding_done = False
    
    while not blob_finding_done:

        ret, image = cap.read()
        if ret:

            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blob_list, blob_image, thresh_image = blob_finder.find(gray_image)

            if len(blob_list) == 1:
                blob_capture_count += 1
                if blob_capture_count >= required_blob_capture_count:
                    blob = blob_list[0]
                    cv2.imshow('camera', image)
                    cv2.imshow('blob_image', blob_image)
                    blob_finding_done = True

            try_count += 1
            if try_count > max_try_count and not blob_finding_done:
                blob_finding_done = True

        time.sleep(0.10)

    if blob is not None:
        blob_x = blob['centroid_x']
        blob_y = blob['centroid_y']
        camera_pts.append((blob_x, blob_y))
        projector_pts.append((dot_x, dot_y))
    
    pt_count += 1
    if pt_count >= len(ind_list):
        collection_done = True


camera_pts = np.array(camera_pts)
projector_pts = np.array(projector_pts)
homography_matrix, mask = cv2.findHomography(camera_pts, projector_pts, cv2.RANSAC)

camera_pts_3d = np.ones((camera_pts.shape[0],3))
camera_pts_3d[:,:2] = camera_pts

projector_pts_3d = np.ones((projector_pts.shape[0],3))
projector_pts_3d[:,:2] = projector_pts

homography_matrix_t = homography_matrix.transpose()
projector_pts_3d_pred = np.dot(camera_pts_3d, homography_matrix_t)

denom = np.zeros((projector_pts.shape[0],2))
denom[:,0] = projector_pts_3d_pred[:,2]
denom[:,1] = projector_pts_3d_pred[:,2]
projector_pts_pred = projector_pts_3d_pred[:,:2]/denom

print(projector_pts.shape)
error = (projector_pts - projector_pts_pred)**2
print(error.shape)
error = error.sum(axis=1)
print(error.shape)
error = np.sqrt(error)
print(error.shape)
error = error.mean()
print(error.shape)

print(f'reprojection error: {error}')

proj_image = np.zeros((proj_monitor.height, proj_monitor.width, 3) ,dtype=np.uint8)
for pt_true, pt_pred in zip(projector_pts, projector_pts_pred):
    proj_image = cv2.circle(proj_image, (int(pt_true[0]), int(pt_true[1])), 10, (255,255,255), cv2.FILLED, cv2.LINE_8,0)
    proj_image = cv2.circle(proj_image, (int(pt_pred[0]), int(pt_pred[1])), 5, (  0,  0,255), cv2.FILLED, cv2.LINE_8,0)
    
cv2.imshow('projector', proj_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
cap.release()
