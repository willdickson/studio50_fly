import cv2
import time
import screeninfo
import numpy as np


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


def get_monitor_dict():
    monitor_list = screeninfo.get_monitors()
    print(monitor_list)
    monitor_dict = {}
    for item in monitor_list:
        monitor_dict[item.name] = item
    return monitor_dict


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    num_ray = 20
    ray_color = (255,255,255)
    angle_rate = -20.0
    sleep_t = 0.001


    monitor_dict = get_monitor_dict()
    monitor = monitor_dict['DP-1']


    cv2.namedWindow('image')
    win_x = monitor.x 
    win_y = monitor.y
    cv2.moveWindow('image', monitor.x, monitor.y)
    main_image = np.zeros((monitor.height, monitor.width, 3) ,dtype=np.uint8)

    x0 = monitor.width/2
    y0 = monitor.height/2

    done = False
    angle = 0.0
    t_start = time.time()
    t_last = t_start

    pos_x = monitor.x + monitor.width//2
    pos_y = monitor.y + monitor.height//2

    period = 15.0
    amplitude = 100.0

    t_start = time.time()

    while not done:

        t_now = time.time()
        t_ellapsed = t_now - t_start

        mod_angle = angle % 360.0 
        x0 = monitor.width/2 + amplitude*np.cos(2.0*np.pi*t_ellapsed/period)
        y0 = monitor.height/2 + amplitude*np.sin(2*2.0*np.pi*t_ellapsed/period)
        ray_image = get_ray_image(x0, y0, np.deg2rad(mod_angle),main_image.shape,num_ray,ray_color)
        cv2.imshow('image', ray_image)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            done = True

        while (time.time() - t_last) < sleep_t:
            time.sleep(0.001)

        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now

        angle_step = angle_rate*dt
        angle += angle_step

        print(f'{mod_angle:0.2f}, {dt:0.3f}')

    cv2.destroyAllWindows()
        


