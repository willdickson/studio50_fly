import cv2
import time
import screeninfo
import numpy as np

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def get_monitor_dict():
    monitor_list = screeninfo.get_monitors()
    print(monitor_list)
    monitor_dict = {}
    for item in monitor_list:
        monitor_dict[item.name] = item
    return monitor_dict


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    num_lookup = 5*360 
    use_lookup = False 
    angle_rate = 0.0
    weighted_sum = True 
    sleep_t = 0.005

    monitor_dict = get_monitor_dict()
    monitor = monitor_dict['DP-1']

    t_start = time.time()
    checker_image = cv2.imread('checker_board_640x480.jpg')
    #checker_image = cv2.resize(checker_image, (monitor.height,monitor.width), interpolation=cv2.INTER_AREA)
    checker_image = checker_image[:monitor.height,:monitor.width]

    if use_lookup:
        lookup_table = {}
        angle_array = np.linspace(0.0,360.0,num_lookup+1, endpoint=True)
        delta_angle = angle_array[1] - angle_array[0]
        for i, angle in enumerate(angle_array):
            rotated_image = rotate(checker_image, angle)
            lookup_table[i] = rotated_image

    cv2.namedWindow('image')
    win_x = monitor.x 
    win_y = monitor.y
    cv2.moveWindow('image', win_x, win_y)

    main_image = np.zeros((monitor.height, monitor.width, 3) ,dtype=np.uint8)
    main_nrow, main_ncol, _ = main_image.shape
    stim_nrow, stim_ncol, _ = checker_image.shape

    #row_fudge = -10
    #row0 = main_nrow//2 - stim_nrow//2  + row_fudge
    #row1 = row0 + stim_nrow

    #col_fudge = 20 
    #col0 = main_ncol//2 - stim_ncol//2 + col_fudge
    #col1 = col0 + stim_ncol


    done = False
    angle = 0.0
    t_last = time.time()

    while not done:

        mod_angle = angle % 360.0 

        if use_lookup:
            pos = mod_angle*num_lookup/360.0
            ind_lower = int(np.floor(pos))
            ind_upper = int(np.ceil(pos))

            if weighted_sum:
                image0 = lookup_table[ind_lower]
                image1 = lookup_table[ind_upper]
                w1 = abs(pos - ind_lower)
                w0 = 1.0 - w1
                stim_image = cv2.addWeighted(image0, w0, image1, w1, 0)
            else:
                if abs(pos - ind_lower) < abs(pos - ind_upper):
                    stim_image = lookup_table[ind_lower]
                else:
                    stim_image = lookup_table[ind_upper]
        else:
            stim_image = rotate(checker_image, mod_angle)


        #main_image[row0:row1, col0:col1, :] = stim_image
        #cv2.imshow('image', main_image)
        cv2.imshow('image', stim_image)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            done = True

        while (time.time() - t_last) < sleep_t:
            time.sleep(0.001)
            pass

        t_now = time.time()
        dt = t_now - t_last
        t_last = t_now

        angle_step = angle_rate*dt
        angle += angle_step

        print(f'{mod_angle:0.2f}, {dt:0.3f}')




    cv2.destroyAllWindows()
        


