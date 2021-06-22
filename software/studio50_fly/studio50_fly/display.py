import os
import cv2
import time
import enum
import numpy as np
from .utility import rotate_image
from .utility import get_monitor_dict
from .utility import create_ray_image
from .utility import create_circular_gradient_image


class DisplayMode(enum.IntEnum):

    BLACK = 0
    SOLID = 1
    STATIC_IMAGE = 2
    ROTATING_IMAGE = 3
    ROTATING_RAYS = 4
    FILLED_CIRCLE = 5
    FILLED_CIRCLE_ARRAY = 6
    SOLID_BLINKING = 7
    GRAYSCALE_GRADIENT = 8


class DisplayController:

    def __init__(self, config, images=None):
        self.config = config
        self.param = dict(self.config['projector'])
        if images is not None:
            try:
                self.param['images'].update(images)
            except KeyError:
                self.param['images'] = images
        self.image_dict = {}
        self.load_images()
       
        monitor_dict = get_monitor_dict()
        self.monitor = monitor_dict[self.param['device']]

        self.window_name = 'projector'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.monitor.width, self.monitor.height)
        cv2.moveWindow(self.window_name, self.monitor.x, self.monitor.y)

        self.image_methods = {
                DisplayMode.BLACK               : self.black_image,
                DisplayMode.SOLID               : self.solid_image,
                DisplayMode.STATIC_IMAGE        : self.static_image,
                DisplayMode.ROTATING_IMAGE      : self.rotating_image, 
                DisplayMode.ROTATING_RAYS       : self.rotating_rays_image, 
                DisplayMode.FILLED_CIRCLE       : self.filled_circle,
                DisplayMode.FILLED_CIRCLE_ARRAY : self.filled_circle_array,
                DisplayMode.SOLID_BLINKING      : self.solid_blinking,
                DisplayMode.GRAYSCALE_GRADIENT  : self.grayscale_gradient,
                }

    def load_images(self):
        if 'images' in self.param:
            for key, file_name in self.param['images'].items():
                if not os.path.exists(file_name):
                    raise FileNotFoundError(f'{file_name} does not exist')
                self.image_dict[key] = cv2.imread(file_name)
        else:
            self.image_dict = {}

    def black_image(self):
        color = (0,0,0)
        return self.solid_image(color=color)

    def solid_image(self,color=(255,255,255)):
        image = np.zeros((self.monitor.height, self.monitor.width, 3) ,dtype=np.uint8)
        image[:,:,:] = color
        return image

    def static_image(self,name=None):
        if name is None:
            image = solid_image(color=(255,0,0))
        else:
            image = self.image_dict[name]
        return image

    def rotating_image(self,t=0.0, rate=0.0, center=None, name=None):
        angle = t*rate
        if name is None:
            image = solid_image(color=(255,0,0))
        else:
            image = self.image_dict[name]
        image_rotated = rotate_image(image, angle, center=center) 
        return image_rotated

    def rotating_rays_image(self, t=0.0, pos=(0,0), rate=0.0, num_rays=3, color=(255,255,255)):
        scale = int(self.param['ray_image_scale'])
        x, y = pos
        x_scaled = x//scale
        y_scaled = y//scale
        angle = np.deg2rad(t*rate)
        image_shape = (self.monitor.height//scale, self.monitor.width//scale, 3)
        image = create_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
        return image

    def filled_circle(self, pos=(0,0), size=1, color=(255,255,255)):
        image_shape = self.monitor.height, self.monitor.width, 3
        image = np.zeros(image_shape, dtype=np.uint8)
        image = cv2.circle(image, pos, size, color, cv2.FILLED, cv2.LINE_8,0)
        return image

    def filled_circle_array(self, pos_list=[], size=1, color=(255,255,255), image=None):
        if image is None:
            image_shape = self.monitor.height, self.monitor.width, 3
            image = np.zeros(image_shape, dtype=np.uint8)
        for pos in pos_list:
            image = cv2.circle(image,(int(pos[0]), int(pos[1])), size, color, cv2.FILLED, cv2.LINE_8,0)
        return image

    def solid_blinking(self, t=0.0, period=1.0, duty_cycle=0.5, on_color=(255,255,255), off_color=(0,0,0)):
        t_mod_period = t % period
        if t_mod_period < duty_cycle*period:
            image = self.solid_image(color=on_color)
        else:
            image = self.solid_image(color=off_color)
        return image

    def grayscale_gradient(self, pos, radius):
        image_shape = (self.monitor.height, self.monitor.width, 3)
        image = create_circular_gradient_image(pos[0], pos[1], radius, image_shape)
        return image

    def update_image(self,state,show=True):
        image = self.image_methods[state['mode']](**state['kwargs'])
        if show:
            cv2.imshow(self.window_name, image)
        return image

#
