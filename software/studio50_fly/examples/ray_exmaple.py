import sys
import cv2
import h5py
import json
import numpy as np
from studio50_fly.utility import create_ray_image
from studio50_fly.homography import Homography

def rotating_rays_image(scale, proj_width, proj_height, t=0.0, pos=(0,0), rate=0.0, num_rays=3, color=(255,255,255)):
    x = int(pos[0])
    y = int(pos[1])
    x_scaled = x//scale
    y_scaled = y//scale
    angle = np.deg2rad(t*rate)
    image_shape = (proj_height//scale, proj_width//scale, 3)
    image = create_ray_image(x_scaled, y_scaled, angle, image_shape, num_rays, color=color)
    return image

def create_black_image(scale, proj_width, proj_height):
    image_shape = (proj_height//scale, proj_width//scale, 3)
    return np.zeros(image_shape,dtype=np.uint8)


# -----------------------------------------------------------------------------------------------


line_len = 12
proj_width = 1920
proj_height = 1080

data_filename = sys.argv[1]
if len(sys.argv) > 2:
    vid_filename = sys.argv[2]
else:
    vid_filename = 'ray_video.avi'

data = h5py.File(data_filename, 'r')
attr = json.loads(data.attrs['jsonparam'])
cam_width = attr['config']['camera']['width']
cam_height = attr['config']['camera']['height']
vid_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_writer = cv2.VideoWriter(vid_filename, vid_fourcc, 30.0, (cam_width, cam_height))

scale = 1 
#scale = 3 
config = attr['config']
homography_data = attr['cal_data']['homography']
homography = Homography(config, cal_data=homography_data)
warp_matrix = np.linalg.inv(homography.matrix.transpose())

trial_schedule = attr['param']['schedule']
trial_dict = attr['param']['trials']
num_data_items = len(data['position'])

for i in range(num_data_items):
    print(f'{i+1}/{num_data_items}')
    t_trial = data['t_trial'][i]
    pos = tuple(data['position'][i])
    body_vec = np.array(data['body_vector'][i])
    pos_proj = homography.camera_to_projector(pos)
    trial_num = int(data['trial_num'][i])
    trial_type = trial_schedule[trial_num]
    if trial_type in ('rotation_neg', 'rotation_pos'):
        trial_param = trial_dict[trial_type]
        num_rays = trial_param['num_rays']
        color = trial_param['color']
        rate = trial_param['rate']
        image = rotating_rays_image(scale, proj_width, proj_height, t_trial, pos_proj, rate, num_rays, color)
        circle_color = (255, 255, 255)
    else:
        image = create_black_image(scale, proj_width, proj_height)
        circle_color = (0,0,0)
    if scale != 1:
        image = cv2.resize(image, (proj_width, proj_height), cv2.INTER_LINEAR)
    image = cv2.warpPerspective(image, warp_matrix, (cam_width, cam_height))
    image = cv2.circle(image, pos, int(0.5*line_len), circle_color, -1)
    p = pos + line_len*body_vec
    q = pos - line_len*body_vec
    p = int(p[0]), int(p[1])
    q = int(q[0]), int(q[1])
    image = cv2.line(image, p, q, (0,0,255), 5)

    vid_writer.write(image)
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

vid_writer.release()







