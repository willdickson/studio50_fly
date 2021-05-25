import cv2
import screeninfo
import numpy as np


def get_monitor_dict():
    monitor_list = screeninfo.get_monitors()
    monitor_dict = {}
    for item in monitor_list:
        monitor_dict[item.name] = item
    return monitor_dict


def get_user_monitor(config):
    monitor_dict = get_monitor_dict()
    return monitor_dict[config['monitor']['device']]


def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def create_ray_image(x0, y0, angle, image_shape, num_rays, color=(0,255,0)):
    nrow, ncol, nchan = image_shape
    x_coord = np.arange(ncol, dtype=np.float) - x0
    y_coord = np.arange(nrow, dtype=np.float) - y0
    x_coord, y_coord = np.meshgrid(x_coord, y_coord)
    coord_angles = (np.arctan2(y_coord, x_coord) - angle) % (2.0*np.pi)
    mask = np.full((nrow,ncol),False)
    for i in range(num_rays):
        theta0 = (2.0*np.pi*((2*i)/float(2*num_rays)))  
        theta1 = (2.0*np.pi*((2*i+1)/float(2*num_rays))) 
        mask_arc = np.logical_and(coord_angles >= theta0, coord_angles < theta1)
        mask = np.logical_or(mask, mask_arc)
    #ray_image = np.zeros((nrow, ncol, nchan), dtype=np.float)
    ray_image = np.zeros((nrow, ncol, nchan), dtype=np.uint8)
    ray_image[mask,:] = color
    return ray_image


def get_angle_and_body_vector(moments): 
    """
    Computre the angle and body vector given the image/blob moments
    """
    body_cov = np.array( [ [moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02'] ]])
    eig_vals, eig_vecs = np.linalg.eigh(body_cov)
    max_eig_ind = np.argmax(eig_vals**2)
    max_eig_vec = eig_vecs[:,max_eig_ind]
    angle = np.arctan2(max_eig_vec[1], max_eig_vec[0])
    return angle, max_eig_vec


def get_max_area_blob(blob_list):
    blob_area_array = np.array([blob['area'] for blob in blob_list])
    ind = blob_area_array.argmax()
    return blob_list[ind]
