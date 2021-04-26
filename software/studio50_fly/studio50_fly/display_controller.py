import os
import cv2
import time
import enum
import numpy as np
from .utility import get_monitor_dict

class DisplayMode(enum.Enum):
    BLACK = 0
    SOLID = 1
    STATIC_IMAGE = 2
    ROTATING_IMAGE = 3
    ROTATING_RAYS = 4


class DisplayController:

    def __init__(self,param):
        self.param = param
        self.image_dict = {}
        self.load_images()
       
        monitor_dict = get_monitor_dict()
        self.monitor = monitor_dict[self.param['monitor_name']]

        self.window_name = 'projector'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.monitor.width, self.monitor.height)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)

        self.next_image_table = {
                DisplayMode.BLACK          : self.next_black_image,
                DisplayMode.SOLID          : self.next_solid_image,
                DisplayMode.STATIC_IMAGE   : self.next_static_image,
                DisplayMode.ROTATING_IMAGE : self.next_rotating_image, 
                DisplayMode.ROTATING_RAYS  : self.next_rotating_rays_image, 
                }

    def load_images(self):
        for key, file_name in self.param['images'].items():
            if not os.path.exists(file_name):
                raise FileNotFoundError(f'{file_name} does not exist')
            self.image_dict[key] = cv2.imread(file_name)

    def next_black_image(self):
        color = (0,0,0)
        return self.next_solid_image(color=color)

    def next_solid_image(self,color=(255,255,255)):
        image = np.zeros((self.monitor.height, self.monitor.width, 3) ,dtype=np.uint8)
        image[:,:,:] = color
        return image

    def next_static_image(self,name=None):
        if name is None:
            image = next_solid_image(color=(255,0,0))
        else:
            image = self.image_dict[name]
        return image

    def next_rotating_image(self,t=0.0, rate=0.0, center=None, name=None):
        angle = t*rate
        if name is None:
            image = next_solid_image(color=(255,0,0))
        else:
            image = self.image_dict[name]
        image_rotated = rotate_image(image, angle, center=center) 
        return image_rotated

    def next_rotating_rays_image(self, t=0.0, pos=(0,0),  rate=0.0, num_rays=3, color=(255,255,255)):
        scale = int(self.param['gen_image_scale'])
        x, y = pos
        x_scaled = x//scale
        y_scaled = y//scale
        angle = np.deg2rad(t*rate)
        image_shape = (self.monitor.height//scale, self.monitor.width//scale, 3)
        image = get_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
        return image

    def update_image(self,state):
        next_image_method = self.next_image_table[state['mode']]
        image = next_image_method(**state['kwargs'])
        cv2.imshow(self.window_name, image)


# Utility functions
# ---------------------------------------------------------------------------------------

def rotate_image(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def get_ray_image(x0, y0, angle, image_shape, num_rays, color=(0,255,0)):
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
    ray_image = np.zeros((nrow, ncol, nchan), dtype=np.float)
    ray_image[mask,:] = color
    return ray_image




#
