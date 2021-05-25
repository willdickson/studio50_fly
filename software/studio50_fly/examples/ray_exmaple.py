import sys
import cv2
import h5py
import json
import numpy as np
from studio50_fly.utility import create_ray_image
from studio50_fly.homography import Homography

def rotating_rays_image(scale, mon_width, mon_height, t=0.0, pos=(0,0), rate=0.0, num_rays=3, color=(255,255,255)):
    x, y = pos
    x_scaled = x//scale
    y_scaled = y//scale
    angle = np.deg2rad(t*rate)
    image_shape = (mon_height//scale, mon_width//scale, 3)
    image = create_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
    return image

def create_black_image(scale, mon_width, mon_height):
    image_shape = (mon_height//scale, mon_width//scale, 3)
    return np.zeros(image_shape,dtype=np.uint8)


filename = sys.argv[1]
data = h5py.File(filename, 'r')
attr = json.loads(data.attrs['jsonparam'])

# projector size
mon_width = 1920
mon_height = 1080
cam_width = attr['config']['camera']['width']
cam_height = attr['config']['camera']['height']

scale = attr['config']['projector']['ray_image_scale']
config = attr['config']
homography_data = attr['cal_data']['homography']
homography = Homography(config, cal_data=homography_data)
warp_matrix = np.linalg.inv(homography.matrix.transpose())

trial_schedule = attr['param']['schedule']
trial_dict = attr['param']['trials']

num_data_items = len(data['position'])

for i in range(num_data_items):
    t_trial = data['t_trial'][i]
    pos = tuple(data['position'][i])
    pos_proj = homography.camera_to_projector(pos)
    trial_num = int(data['trial_num'][i])
    trial_type = trial_schedule[trial_num]
    if trial_type in ('rotation_neg', 'rotation_pos'):
        trial_param = trial_dict[trial_type]
        num_rays = trial_param['num_rays']
        color = trial_param['color']
        rate = trial_param['rate']
        image = rotating_rays_image(scale, mon_width, mon_height, t_trial, pos_proj, rate, num_rays, color)
    else:
        image = create_black_image(scale, mon_width, mon_height)
    image = cv2.resize(image, (mon_width, mon_height), cv2.INTER_LINEAR)
    image = cv2.warpPerspective(image, warp_matrix, (cam_width, cam_height))
    image = cv2.circle(image, pos, 10, (0,0,255), 2)
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break







