import cv2
import time
import numpy as np
from studio50_fly import DisplayMode
from studio50_fly import DisplayController

param = {
        'monitor_name'    : 'DP-1',
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

    # Set state for image update
    if 0:
        state = {'mode': DisplayMode.BLACK}
    if 0:
        state = {
                'mode': DisplayMode.SOLID,
                'kwargs': {'color': (255, 100, 100)}
                }

    if 0:
        state = {
                'mode': DisplayMode.IMAGE, 
                'kwargs': {'name' : 'checker_20x20_black_white'}
                }

    if 1:
        state = {
                'mode' : DisplayMode.ROTATING_IMAGE, 
                'kwargs' : { 
                    't'    : t, 
                    'rate' : 10.0,
                    'name' : 'checker_20x20_black_white',
                    }
                }

    if 0:
        x = disp.monitor.width//2 + 100*np.cos(2.0*np.pi*t/15.0)  
        y = disp.monitor.height//2 + 100*np.sin(4.0*np.pi*t/15.0)
        state = {
                'mode'  : DisplayMode.ROTATING_RAYS, 
                'kwargs' : {
                    't'   : t, 
                    'pos' : (x,y),
                    'rate': 20.0, 
                    'num_rays' : 15,
                    'color': (255,255,255),
                    }
                }

    disp.update_image(state)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        done = True

