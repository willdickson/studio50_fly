import os
import cv2
import time
import numpy as np
from .utility import get_monitor_dict


class DisplayController:

    def __init__(self,param):
        self.param = param
        self.image_dict = {}
        self.load_images()
       
        # Get monitor info
        monitor_dict = get_monitor_dict()
        self.monitor = monitor_dict[self.param['monitor_name']]

        # Setup named window for display
        self.window_name = 'projector'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.monitor.width, self.monitor.height)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)


    def load_images(self):
        for key, file_name in self.param['images'].items():
            if not os.path.exists(file_name):
                raise FileNotFoundError(f'{file_name} does not exist')
            self.image_dict[key] = cv2.imread(file_name)


    def next_image(self,state={}):
        if state['mode'] == 'black':
            image = self.solid_image((0,0,0))
        elif state['mode'] == 'solid':
            image = self.solid_image(state['vals']['color'])
        elif state['mode'] == 'image':
            image = self.image_dict[state['vals']['name']]
        elif state['mode'] == 'rotating_image':
            image = self.next_rotating_image(state['vals'])
        elif state['mode'] == 'rotating_rays':
            image = self.next_rotating_rays_image(state['vals'])
        else:
            raise ValueError(f"unknown mode {state['mode']}")
        return image


    def solid_image(self,color):
        image = np.zeros((self.monitor.height, self.monitor.width, 3) ,dtype=np.uint8)
        image[:,:,:] = color
        return image


    def next_rotating_image(self,vals):
        t = vals['t']
        rate = vals['rate']
        angle = t*rate
        image = self.image_dict[vals['name']]
        image_rotated = rotate_image(image, angle) 
        return image_rotated


    def next_rotating_rays_image(self,vals):
        num_rays = vals['num']
        color = vals['color']
        scale = int(self.param['ray_image_scale'])
        x_scaled = vals['x']//scale
        y_scaled = vals['y']//scale
        angle = np.deg2rad(vals['t']*vals['rate'])
        image_shape = (self.monitor.height//scale, self.monitor.width//scale, 3)
        image = get_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
        return image


    def update_image(self,state):
        image = self.next_image(state)
        cv2.imshow(self.window_name, image)


# Utility functions
# ---------------------------------------------------------------------------------------

def rotate_image(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def get_ray_image(x0, y0, angle, image_shape, num_rays, color=(0,255,0)):
    nrow_orig, ncol_orig, nchan = image_shape
    nrow = nrow_orig
    ncol = ncol_orig
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
    ray_image = np.zeros((nrow, ncol, 3), dtype=np.float)
    ray_image[mask,:] = color
    return ray_image




#
