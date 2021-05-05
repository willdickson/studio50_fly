import os
import cv2
import time
import pickle
import numpy as np
from .config import Config
from .camera import Camera
from .display import DisplayMode
from .display import DisplayController
from .homography import Homography
from .blob_finder import BlobFinder
from .utility import get_user_monitor


class Calibration:

    def __init__(self,config):
        self.config = config
        self.load_homography_cal()
        self.load_position_cal()

    def load_homography_cal(self):
        self.homography = Homography(self.config)

    def load_position_cal(self):
        filename = self.config['calibration']['position']['filename']
        filename = os.path.join(self.config.path,filename)
        with open(filename,'rb') as f:
            self.arena = pickle.load(f)

    def homography_data(self,jsonable=True):
        return dict(self.homography.calibration_data(jsonable=jsonable))

    def position_data(self,jsonable=True):
        position_data = dict(self.arena)
        if jsonable:
            del position_data['contour_image']
            for k,v in position_data.items():
                if type(v) == np.ndarray:
                    position_data[k] = v.tolist()
        return position_data

    def data(self,jsonable=True):
        data = {
                'position'   : self.position_data(jsonable=jsonable),
                'homography' : self.homography_data(jsonable=jsonable),
                }
        return data



# Calibration procedures
# ---------------------------------------------------------------------------------------  

def run_homography_calibration():

    print()
    print(' homography calibration')
    print(' ======================')
    print()
    print(' make sure IR light is off and the IR passfilter is removed') 
    print()
    print(' press enter to continue',end='')
    print()
    ans = input()

    window_name = 'calibration'
    config = Config()
    homography_config = config['calibration']['homography']

    camera = Camera(config,'calibration-homography')
    display = DisplayController(config)
    user_monitor = get_user_monitor(config)

    blob_finder = BlobFinder(**homography_config['blob_finder'])

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name,config['camera']['width'], config['camera']['height'])
    cv2.moveWindow(window_name, user_monitor.width - config['camera']['width'], 0)

    cal_data = {'camera_pts' : [], 'projector_pts': []}

    # Create test points for homography calibration
    mid_x = display.monitor.width//2
    mid_y = display.monitor.height//2
    min_x, max_x = homography_config['x_range']
    min_y, max_y = homography_config['y_range']
    step = homography_config['step_size']
    test_pts = [(x+mid_x, y+mid_y) for x in range(min_x,max_x,step) for y in range(min_y,max_y,step)]

    aborted = False
    collection_done = False
    collection_count = 0

    # Loop over test points and find associated blobs in camera image 
    for pt in test_pts:

        display_state = {
                'mode': DisplayMode.FILLED_CIRCLE, 
                'kwargs': {
                    'pos'   : pt,
                    'size'  : homography_config['circle_size'],
                    'color' : (255, 255, 255),
                    }
                }
        display.update_image(display_state)

        key = cv2.waitKey(100) & 0xff
        if key == ord('q'): 
            print()
            print(' calibration aborted')
            print()
            aborted = True
            break

        attempt_count = 0
        blob_found_count = 0
        blob_finding_done = False
        
        # Attempt to find blobs in image 
        while not blob_finding_done:
            ok, image = camera.read()
            if ok:
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                blob_list, blob_image, thresh_image = blob_finder.find(gray_image)

                if len(blob_list) == 1:
                    blob = blob_list[0]
                    blob_found_count += 1
                    if blob_found_count >= homography_config['min_required']:
                        cv2.imshow(window_name, blob_image)
                        blob_finding_done = True
                        blob_x = blob['centroid_x']
                        blob_y = blob['centroid_y']
                        cal_data['camera_pts'].append((blob_x, blob_y))
                        cal_data['projector_pts'].append(pt)
                        print(f'  (cx,cy) = {blob_x}, {blob_y}')

                attempt_count += 1
                if not blob_finding_done:
                    if attempt_count > homography_config['max_attempts']:  
                        blob_finding_done = True

            time.sleep(homography_config['capture_dt'])

    if not aborted:

        homography = Homography(config,cal_data)
        error = homography.reprojection_error()
        projector_pts_pred = homography.reproject_cal_data()

        print()
        print(f'  reprojection error: {error:0.3}')
        print()

        display_state = {
                'mode' : DisplayMode.FILLED_CIRCLE_ARRAY, 
                'kwargs' : {
                    'pos_list' : cal_data['projector_pts'],
                    'size' : 10,
                    'color' : (255,255,255),
                    }
                }
        image = display.update_image(display_state)
        display_state = {
                'mode' : DisplayMode.FILLED_CIRCLE_ARRAY, 
                'kwargs' : {
                    'pos_list' : projector_pts_pred,
                    'size' :  5,
                    'color' : (0,0,255),
                    'image' : image,
                    }
                }
        display.update_image(display_state)

        print(' save calibration data (y=yes, n=no)')
        save_dialog_done = False
        while not save_dialog_done:
            key = cv2.waitKey(0) & 0xff
            if (key == ord('y')):
                print(f' saving calibration to: {homography.calibration_file}')
                homography.save_calibration()
                save_dialog_done = True
            elif (key == ord('n')):
                print(' calibration not saved!')
                save_dialog_done = True
            else:
                print(' please enter y or n')
        print()

    camera.release()
    cv2.destroyAllWindows()


def run_position_calibration():
    print()
    print(' position calibration (press q when done)')
    print(' ========================================')
    print()

    window_name = 'calibration'

    config = Config()

    camera = Camera(config, 'calibration-position')
    display = DisplayController(config)
    user_monitor = get_user_monitor(config)

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name,config['camera']['width'], config['camera']['height'])
    cv2.moveWindow(window_name, user_monitor.width - config['camera']['width'], 0)

    display_state = {'mode': DisplayMode.SOLID, 'kwargs': {'color': (255, 255, 255)}}
    display.update_image(display_state)
    cv2.waitKey(10)

    done = False
    while not done:
        ok, frame = camera.read()
        if ok:
            arena_data = find_arena(frame, threshold=config['calibration']['position']['threshold'])
            cv2.imshow(window_name, arena_data['contour_image'])
    
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'): 
            done = True
    
    display_state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
    display.update_image(display_state)

    print(' save calibration data (y=yes, n=no)')
    save_dialog_done = False
    while not save_dialog_done:
        key = cv2.waitKey(0) & 0xff
        if (key == ord('y')):
            filename = os.path.join(config.path,config['calibration']['position']['filename'])
            print(f' saving calibration to: {filename}')
            with open(filename, 'wb') as f:
                pickle.dump(arena_data,f)
            save_dialog_done = True
        elif (key == ord('n')):
            print(' calibration not saved!')
            save_dialog_done = True
        else:
            print(' please enter y or n')
    print()

    camera.release()
    cv2.destroyAllWindows()


def find_arena(image, threshold=100):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    rval, threshold_image = cv2.threshold(gray_image, threshold, np.iinfo(image.dtype).max, cv2.THRESH_BINARY)
    contour_list, dummy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area_array = np.array([cv2.contourArea(c) for c in contour_list])
    if area_array.size < 1:
        contour_data = {
                'area' : 0,
                'contour': None,  
                'centroid_x': 0,
                'centroid_y': 0,
                'contour_image': image,
                'bounding_box' : None,
                }
    else:
        ind_of_max = area_array.argmax()
        area = area_array[ind_of_max]
        contour = contour_list[ind_of_max]
        moments = cv2.moments(contour) 
        centroid_x = int(np.round(moments['m10']/moments['m00']))
        centroid_y = int(np.round(moments['m01']/moments['m00']))
        bounding_box = cv2.boundingRect(contour)

        contour_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_image,[contour],-1,(0,0,255),2)
        cv2.circle(contour_image,(int(centroid_x),int(centroid_y)),10,(255,0,),2)
        bbx, bby, bbw, bbh = bounding_box
        cv2.rectangle(contour_image,(bbx,bby),(bbx+bbw,bby+bbh),(0,255,0),2)

        contour_data = {
                'area' : area,
                'contour': contour,  
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'contour_image': contour_image,
                'bounding_box' : bounding_box,
                }
    return contour_data
