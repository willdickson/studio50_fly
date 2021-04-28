import cv2
import time
import numpy as np
from studio50_fly import DisplayMode
from studio50_fly import DisplayController

param = {
        'device'    : 'DP-1',
        'gen_image_scale' :  3, 
        'images': {
            'checker_20x20_black_green' : 'checker_20x20_black_green_640x360.jpg',
            'checker_20x20_black_white' : 'checker_20x20_black_white_640x360.jpg',
            'spiral_r10_black_white'    : 'spiral_r10_black_white_640x360.jpg',
            },
        }

disp = DisplayController(param)

t_start = time.time()
t_last = t_start

done = False

while not done:

    # Update time information
    t_now = time.time()
    dt = t_now - t_last
    t  = t_now - t_start
    t_last = t_now

    if t <= 2.0: 
        display_state = {'mode': DisplayMode.BLACK, 'kwargs': {}} 
    elif t > 2.0 and t <= 4.0:
        display_state = {
                'mode': DisplayMode.STATIC_IMAGE, 
                'kwargs': {'name' : 'checker_20x20_black_white'}
                }
    elif t > 4.0 and t <= 5.0: 
        display_state = {'mode': DisplayMode.BLACK, 'kwargs': {}} 
    elif t > 5.0 and t <= 15.0:
       x = disp.monitor.width//2 + 100*np.cos(2.0*np.pi*t/15.0)  
       y = disp.monitor.height//2 + 100*np.sin(4.0*np.pi*t/15.0)
       display_state = {
               'mode'  : DisplayMode.ROTATING_RAYS, 
               'kwargs' : {
                   't'   : t, 
                   'pos' : (x,y),
                   'rate': 20.0, 
                   'num_rays' : 15,
                   'color': (255,255,255),
                   }
               }
    elif t > 15.0 and t <= 16.0: 
        display_state = {'mode': DisplayMode.BLACK, 'kwargs': {}} 
    elif t > 16.0 and t <= 20.0:
        display_state = {
                'mode': DisplayMode.STATIC_IMAGE, 
                'kwargs': {'name' : 'checker_20x20_black_white'}
                }
    if t > 20.0:
        done = True

    disp.update_image(display_state)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        done = True

