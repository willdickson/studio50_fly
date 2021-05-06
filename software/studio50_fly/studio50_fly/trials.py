import os
import cv2
import json
import time
import numpy as np
from h5_logger import H5Logger
from .config import Config
from .camera import Camera
from .utility import get_user_monitor
from .utility import get_angle_and_body_vector
from .blob_finder import BlobFinder
from .homography import Homography
from .calibration import Calibration
from .display import DisplayMode
from .display import DisplayController


class Trials:

    POS_NOT_FOUND = (-1.0, -1.0)

    def __init__(self, param_file, data_file):
        self.param = None
        self.logger = None
        self.bg_image = None
        self.t_start = 0.0
        self.files = {'param': param_file, 'data': data_file}
        self.load_param_file()
        self.config = Config()
        self.camera = Camera(self.config, 'fly')
        self.calibration = Calibration(self.config)
        self.user_monitor = get_user_monitor(self.config)
        self.display = DisplayController(self.config, images=self.param['images'])
        self.create_camera_window()
        self.blob_finder = BlobFinder(**self.config['fly']['blob_finder'])
        self.blob_finder.mask = self.arena_mask()
        self.zero_bg_image()

        self.show_threshold_image = False

    def load_param_file(self):
        if not os.path.exists(self.files['param']):
            raise FileNotFoundError(f"param file not found: {self.files['param']}")
        with open(self.files['param'], 'r') as f:
            self.param = json.load(f)

    def check_data_file(self):
        if os.path.exists(self.files['data']):
            print()
            print(' data file already exists overwrite (y/n): ', end='')
            ans = input()
            print()
            if not ans == 'y':
                print(' ok ...  aborting run')
                print()
                exit(0)

    def create_camera_window(self):
        self.window_name = "studio50 file trials"
        cv2.namedWindow(self.window_name)
        cv2.resizeWindow(
                self.window_name, 
                self.config['camera']['width'], 
                self.config['camera']['height']
                )
        window_pos_x = self.user_monitor.width - self.config['camera']['width']
        window_pos_y = 0
        cv2.moveWindow(self.window_name, window_pos_x, window_pos_y)


    def create_threshold_image(self):
        cv2.namedWindow('threshold')
        cv2.resizeWindow(
                'threshold', 
                self.config['camera']['width'], 
                self.config['camera']['height']
                )
        window_pos_x = self.user_monitor.width - self.config['camera']['width']
        window_pos_y = 0
        cv2.moveWindow('threshold', window_pos_x, window_pos_y)

    def run_attributes(self):
        attributes = {
                'param'    : self.param,
                'config'   : self.config.data,
                'cal_data' : self.calibration.data(jsonable=True) 
                }
        return attributes 

    def run(self):

        print()
        print(f" running studio50 fly")
        print(f" ====================")
        print()
        print(f" param:  {self.files['param']}")
        print(f" output: {self.files['data']}")
        print()
        
        self.check_data_file()
        self.logger = H5Logger(self.files['data'],jsonparam=self.run_attributes())

        state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
        self.display.update_image(state)
        cv2.waitKey(self.config['projector']['start_dt_ms'])

        self.find_bg_image()

        if self.show_threshold_image:
            self.create_threshold_image()

        self.run_trial_schedule()

    def run_trial_schedule(self):
        print(f' running trials (press q to quit)')
        print()
        self.t_start = time.time()
        for cycle_num in range(self.param['cycles']):
            print(f"  cycle: {cycle_num+1}/{self.param['cycles']}")
            print()
            for trial_num, trial_name in enumerate(self.param['schedule']):
                self.run_trial(trial_num, trial_name)
            print()

    def run_trial(self, trial_num, trial_name): 
        t_trial = time.time()
        trial_param = self.param['trials'][trial_name]
        len_schedule = len(self.param['schedule'])
        print(f'   trial {trial_num+1}/{len_schedule}: {trial_name}')

        t_now  = t_trial
        pos = self.POS_NOT_FOUND 
        body_angle = 0.0
        body_vector = np.array([0.0, 0.0])

        while t_now - t_trial < trial_param['duration']:

            t_now = time.time()
            t_elapsed_trial = t_now - t_trial
            t_elapsed_total = t_now - self.t_start

            found = False
            ok, image = self.camera.read()

            if ok:
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                diff_image = cv2.absdiff(gray_image, self.bg_image) 
                blob_list, blob_image, thresh_image = self.blob_finder.find(diff_image)

                if self.show_threshold_image:
                    cv2.imshow('threshold', thresh_image)

                if blob_list:
                    found = True
                    fly = get_max_area_blob(blob_list)
                    pos = (fly['centroid_x'], fly['centroid_y'])
                    body_angle, body_vector = get_angle_and_body_vector(fly['moments']) 
                    self.add_fly_line_to_image(image, (fly['centroid_x'],fly['centroid_y']), body_vector)
                    cv2.circle(
                            image,
                            (int(pos[0]),int(pos[1])),
                            self.config['fly']['circle']['radius'],
                            self.config['fly']['circle']['color'],
                            self.config['fly']['circle']['thickness']
                            )
                cv2.imshow(self.window_name, image)

                log_image_type =  self.config['fly']['log']['image']['type']
                if log_image_type == 'full':
                    log_image = gray_image
                elif log_image_type == 'arena':
                    bbx, bby, bbw, bbh = self.calibration.arena['bounding_box']
                    log_image = gray_image[bby:bby+bbh,bbx:bbx+bbw]
                elif log_image_type == 'fly':
                    image_size = self.config['fly']['log']['image']['fly_image_size']
                    raise ValueError(f'log image type {log_image_type} not implemented yet')
                else:
                    raise ValueError(f'unknown log image type {log_image_type}')

                display_mode = self.update_display(t_elapsed_trial, pos, trial_param)
                data = {
                        'found'        : found,
                        't_total'      : t_elapsed_total,
                        't_trial'      : t_elapsed_trial,
                        'position'     : pos,
                        'body_angle'   : body_angle,
                        'body_vector'  : body_vector,
                        'display_mode' : display_mode,
                        'trial_num'    : trial_num,
                        'image'        : log_image,
                        }
                self.logger.add(data)

    def update_display(self, t, pos, trial_param): 
        display_mode = DisplayMode[trial_param['display_mode'].upper()] 
        if display_mode == DisplayMode.BLACK:
            kwargs = {}
        elif display_mode == DisplayMode.STATIC_IMAGE:
            kwargs = {'name': trial_param['name']} 
        elif display_mode == DisplayMode.ROTATING_RAYS:
            if (trial_param['center'] == 'arena') or (pos == self.POS_NOT_FOUND): 
                cx_arena = self.calibration.arena['centroid_x']
                cy_arena = self.calibration.arena['centroid_y']
                pos = (cx_arena, cy_arena)
            pos_proj = self.calibration.homography.camera_to_projector(pos)
            kwargs = {
                    't'        :   t,
                    'pos'      :   tuple(pos_proj),
                    'rate'     :   trial_param['rate'], 
                    'num_rays' :   trial_param['num_rays'], 
                    'color'    :   trial_param['color'],
                    }
        else:
            raise ValueError(f"unknown display mode {trial_param['display_mode']}")
        self.display.update_image({'mode': display_mode, 'kwargs': kwargs})
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            print()
            print(' run aborted!')
            print()
            exit(0)
        return display_mode

    def add_fly_line_to_image(self,image,pos,vector): 
        s0 = self.config['fly']['circle']['radius']
        s1 = s0 + self.config['fly']['line']['length']
        cx, cy = pos
        for sign in (1,-1):
            pt0 = int(cx + sign*s0*vector[0]), int(cy + sign*s0*vector[1])
            pt1 = int(cx + sign*s1*vector[0]), int(cy + sign*s1*vector[1])
            cv2.line(
                    image, 
                    pt0, 
                    pt1, 
                    self.config['fly']['line']['color'], 
                    self.config['fly']['line']['thickness']
                    )

    def find_bg_image(self):
        print(f' finding background image (press q when done)')
        self.zero_bg_image()
        cv2.imshow(self.window_name, self.bg_image)
        cv2.waitKey(1)

        cnt = 0
        done = False
        while not done:
            ok, image = self.camera.read()
            if ok:
                if cnt > self.config['fly']['background']['min_count']:
                    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    self.bg_image = np.maximum(self.bg_image, gray_image)
                    cv2.imshow(self.window_name, self.bg_image)
                    key = cv2.waitKey(1) & 0xff
                    if key == ord('q'):
                        done = True
                cnt += 1
        print()

    def zero_bg_image(self):
        shape = self.config['camera']['height'], self.config['camera']['width']
        self.bg_image = np.zeros(shape, dtype=np.uint8)

    def arena_mask(self):
        shape = self.config['camera']['height'], self.config['camera']['width']
        arena_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(arena_mask, pts=[self.calibration.arena['contour']], color=(255,))
        kernel_size = self.config['fly']['arena_mask']['kernel_size']
        kernel = np.ones((kernel_size, kernel_size))
        arena_mask = cv2.erode(arena_mask,kernel,iterations=1)
        return arena_mask


# ------------------------------------------------------------------------------------------------

def get_max_area_blob(blob_list):
    blob_area_array = np.array([blob['area'] for blob in blob_list])
    ind = blob_area_array.argmax()
    return blob_list[ind]

