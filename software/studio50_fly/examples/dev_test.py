import cv2
import time
import numpy as np
from studio50_fly import DisplayController

param = {
        'monitor_name'    : 'DP-1',
        'ray_image_scale' :  3, 
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

    # Set state for image update
    if 0:
        state = {'mode': 'black', 'vals': {}}
    if 0:
        state = {
                'mode': 'solid',
                'vals': {
                    'color': (255, 100, 100),
                    }
                }

    if 0:
        state = {
                'mode': 'image', 
                'vals': {
                    'name' : 'checker_20x20_black_green',
                    #'name' : 'checker_20x20_black_white',
                    #'name' : 'spiral_r10_black_white',
                    }
                }

    if 0:
        state = {
                'mode' : 'rotating_image', 
                'vals' : { 
                    't'    : t, 
                    'rate' : 10.0,
                    'name' : 'checker_20x20_black_white',
                    }
                }

    if 1:
        state = {
                'mode'  : 'rotating_rays', 
                'vals' : {
                    't'   : t, 
                    'x'   : disp.monitor.width//2 + 100*np.cos(2.0*np.pi*t/15.0), 
                    'y'   : disp.monitor.height//2 + 100*np.sin(4.0*np.pi*t/15.0),
                    'rate': 10.0, 
                    'num' : 15,
                    'color': (255,0,0),
                    }
                }

        disp.update_image(state)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            done = True

