import os
import cv2
import time
import enum
import numpy as np
from .utility import rotate_image
from .utility import create_ray_image
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

        self.next_image_methods = {
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
        image = create_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
        return image

    def update_image(self,state):
        image = self.next_image_methods[state['mode']](**state['kwargs'])
        cv2.imshow(self.window_name, image)

#
