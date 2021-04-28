import cv2
import time
from .config import Config
from .camera import Camera
from .utility import get_user_monitor
from .blob_finder import BlobFinder
from .homography import Homography
from .display_controller import DisplayMode
from .display_controller import DisplayController



def run_homography_calibration():

    print()
    print('* running homography calibration')
    print()
    print('* make sure IR filter is removed from arena and press enter to continue (q=quit)')
    ans = input()
    if ans == 'q':
        exit(0)

    window_name = 'homography calibration'

    config = Config()

    camera = Camera(config['camera'])
    display = DisplayController(config['projector'])
    user_monitor = get_user_monitor(config['monitor'])

    blob_finder = BlobFinder(**config['calibration']['blob_finder'])

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name,config['camera']['width'], config['camera']['height'])
    cv2.moveWindow(window_name, user_monitor.width - config['camera']['width'], 0)


    homography_config = config['calibration']['homography']
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
            print('collection aborted')
            print()
            aborted = True
            break

        attempt_count = 0
        blob_found_count = 0
        blob_finding_done = False
        
        # Attempt to find blobs in image 
        while not blob_finding_done:
            rval, image = camera.read()
            if rval:
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                blob_list, blob_image, thresh_image = blob_finder.find(gray_image)

                if len(blob_list) == 1:
                    blob = blob_list[0]
                    blob_found_count += 1
                    if blob_found_count >= homography_config['blob_finder']['min_required']:
                        cv2.imshow(window_name, blob_image)
                        blob_finding_done = True
                        blob_x = blob['centroid_x']
                        blob_y = blob['centroid_y']
                        cal_data['camera_pts'].append((blob_x, blob_y))
                        cal_data['projector_pts'].append(pt)
                        print(f'  blob found: {blob_x}, {blob_y}')

                attempt_count += 1
                if not blob_finding_done:
                    if attempt_count > homography_config['blob_finder']['max_attempts']:  
                        blob_finding_done = True

            time.sleep(homography_config['blob_finder']['capture_dt'])

    if not aborted:

        homography = Homography(config,cal_data)
        error = homography.reprojection_error()
        projector_pts_pred = homography.reproject_cal_data()

        print()
        print(f'reprojection error: {error:0.3}')
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

        print('save calibration data (y=yes, n=no)')
        save_dialog_done = False
        while not save_dialog_done:
            key = cv2.waitKey(0) & 0xff
            if (key == ord('y')):
                homography.save_calibration()
                save_dialog_done = True
            elif (key == ord('n')):
                save_dialog_done = True
            else:
                print('please enter y or n')
        print()

    cv2.destroyAllWindows()
    camera.release()



def run_arena_calibration():
    pass


