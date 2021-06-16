import sys
import cv2
import h5py
import json
import numpy as np
from studio50_fly.utility import create_ray_image
import scipy.signal as sig
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

def create_arena_mask(image_background,scale, proj_width, proj_height, centroid_x, centroid_y, radius): 
     circle_img = np.zeros((proj_height//scale,proj_width//scale), np.uint8)
     cv2.circle(circle_img,(centroid_x,centroid_y),radius,1,thickness=-1)
     masked_data = cv2.bitwise_and(image_background, image_background, mask=circle_img)
     image=masked_data
     return image 
 

def find_total_speed(x_pos,y_pos,time, smoothing_type='Savitzky-Golay'):
    dt = time[1:] - time[:-1]
    if smoothing_type == 'Savitzky-Golay':
        window = 601
        order = 2
        x_pos_smooth = sig.savgol_filter(x_pos, window, order, deriv=0)
        y_pos_smooth = sig.savgol_filter(y_pos, window, order, deriv=0)

    elif smoothing_type == 'butterworth':
        b, a = sig.butter(2,0.005)
        x_pos_smooth = sig.filtfilt(b, a, x_pos, padlen = 150)
        y_pos_smooth = sig.filtfilt(b, a, y_pos, padlen = 150)

    else:
        x_pos_smooth = x_pos
        y_pos_smooth = y_pos

    dx = x_pos_smooth[1:] - x_pos_smooth[:-1]
    dy = y_pos_smooth[1:] - y_pos_smooth[:-1]
    disp = np.sqrt(dx**2 + dy**2)
    speed = disp/dt
    return(speed)

def SchmittTrigger(speed, low_speedT, upper_speedT):
    st = np.zeros(len(speed))+2
    idx_min = np.where(speed<low_speedT)
    st[idx_min[0]]=0
    idx_max = np.where(speed>upper_speedT)
    st[idx_max[0]]=1
    # find values between the two thresholds (called x_blocks)
    x_block = np.zeros(len(speed))
    idxx = np.where(st == 2)
    x_block[idxx[0]] = 1
    #find starts of blocks
    x_blocks = np.append(0,x_block)
    start = np.where((np.diff(x_blocks))==1)
    #find ends of blocks
    x_blocke = np.append(x_block,0)
    end = np.where((np.diff(x_blocke))==-1)
    state_before = 0
    for i in range(len(start[0])):
        st[start[0][i]:end[0][i]+1] = state_before
        if end[0][i]+1<len(speed):
            state_before = st[end[0][i]+1]
    return(st)
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
cal_data = attr['cal_data']
arena_contour = np.array(cal_data['position']['contour'])
x_arena = arena_contour[:,0,0]
y_arena = arena_contour[:,0,1]
#find arena center and radius 
arena_center_x = cal_data['position']['centroid_x']
arena_center_y = cal_data['position']['centroid_y']
radius = int((np.max(x_arena) - np.min(x_arena))/2)
center2_x=np.max(x_arena) - radius
center2_y=np.max(y_arena) - radius

fly_line_x = np.array([])
fly_line_y = np.array([])

print(arena_center_x)
print(arena_center_y)
print(center2_x)
print(center2_y)
print(radius)
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
    x_pos = data['position'][i,0]
    y_pos = data['position'][i,1]
    
    body_vec = np.array(data['body_vector'][i])
    pos_proj = homography.camera_to_projector(pos)
    trial_num = int(data['trial_num'][i])
    trial_type = trial_schedule[trial_num]
    if trial_type in ('rotation_neg', 'rotation_pos','static_ray'):
        trial_param = trial_dict[trial_type]
        num_rays = trial_param['num_rays']
        color = trial_param['color']
        rate = trial_param['rate']
        image_background = rotating_rays_image(scale, proj_width, proj_height, t_trial, pos_proj, rate, num_rays, color)
        image = create_arena_mask(image_background,scale, proj_width, proj_height, 910,556,438)
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
    
    fly_line_y = np.append(fly_line_y, y_pos)
    fly_line_x = np.append(fly_line_x,x_pos)
    
    image = cv2.line(image, p, q, (0,0,255), 5)
    curve = np.column_stack((x_arena.astype(np.int32), y_arena.astype(np.int32)))
    image = cv2.polylines(image, [curve], False, (255,255,255))
    fly_line = np.column_stack((fly_line_x.astype(np.int32), fly_line_y.astype(np.int32)))
    image = cv2.polylines(image, [fly_line], False, (255,128,0), thickness=4)
    #mask = np.zeros(image.shape[:2], dtype="uint8")
    #cv2.circle(mask, (arena_center_x, arena_center_y), radius, 255, -1)
    #image = cv2.bitwise_and(image, image, mask=mask)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    bottomLeftCornerOfText = (10,40)

    image=cv2.putText(image,'Stimulus: '+ str(trial_type), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    

    vid_writer.write(image)
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

vid_writer.release() 








