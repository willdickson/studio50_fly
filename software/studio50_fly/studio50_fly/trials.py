import os
import cv2
import json
import time
import numpy as np
from .config import Config
from .camera import Camera
from .utility import get_user_monitor
from .blob_finder import BlobFinder
from .homography import Homography
from .calibration import Calibration
from .display_controller import DisplayMode
from .display_controller import DisplayController


class Trials:

    def __init__(self, param_file, data_file):
        self.param = None
        self.t_start = 0.0
        self.files = {'param': param_file, 'data': data_file}
        self.load_param_file()
        self.config = Config()
        self.camera = Camera(self.config)
        self.calibration = Calibration(self.config)
        self.user_monitor = get_user_monitor(self.config)
        self.display = DisplayController(self.config, images=self.param['images'])
        self.create_camera_window()
        self.zero_bg_image()

    def load_param_file(self):
        if not os.path.exists(self.files['param']):
            raise FileNotFoundError(f"param file not found: {self.files['param']}")
        with open(self.files['param'], 'r') as f:
            self.param = json.load(f)

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

    def set_display_black(self, wait_ms=None):
        state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
        self.display.update_image(state)
        if wait_ms is not None:
            cv2.waitKey(wait_ms)

    def run(self):

        print()
        print(f" running studio50 fly")
        print(f" ====================")
        print()
        print(f" param:  {self.files['param']}")
        print(f" output: {self.files['data']}")
        print()

        self.set_display_black(wait_ms=100)
        self.find_bg_image()
        self.run_trial_schedule()

    def run_trial_schedule(self):
        print(f' running trials (press q to quit)')
        print()
        self.t_start = time.time()
        for cycle_num in range(self.param['cycles']):
            print(f'  cycle: {cycle_num}')
            print()
            for trial_name in self.param['schedule']:
                self.run_trial(trial_name)


    def run_trial(self,trial_name): 
        t_trial = time.time()
        trial_param = self.param['trials'][trial_name]
        print(f'   trial: {trial_name}, {trial_param}')
        t_now = t_trial
        while t_now - t_trial < trial_param['duration']:
            t_now = time.time()
            #self.update_display(trial_param)

    #def update_display(self, trial_param): 
    #    disiplay_mode = DisplayMode[trial_param['display_mode'].upper()] 
    #    if display_mode == DisplayMode.BLACK:
    #        pass
    #    elif display_mode == DisplayMode.STATIC_IMAGE:
    #        pass
    #    elif dsiplay_mode == d


    def find_bg_image(self):
        print(f' finding background image (press q when done)')
        self.zero_bg_image()
        cv2.imshow(self.window_name, self.bg_image)
        cv2.waitKey(1)

        done = False
        while not done:
            ok, image = self.camera.read()
            if ok:
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                self.bg_image = np.maximum(self.bg_image, gray_image)
                cv2.imshow(self.window_name, self.bg_image)
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    done = True
        print()

    def zero_bg_image(self):
        shape = self.config['camera']['height'], self.config['camera']['width']
        self.bg_image = np.zeros(shape, dtype=np.uint8)

def run_trials(param_file, data_file):
    trials = Trials(param_file, data_file)
    trials.run()








    



