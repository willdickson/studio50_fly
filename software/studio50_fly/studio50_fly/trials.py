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
from .display import DisplayMode
from .display import DisplayController


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
        self.blob_finder = BlobFinder(**self.config['fly']['blob_finder'])
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


    def run(self):

        print()
        print(f" running studio50 fly")
        print(f" ====================")
        print()
        print(f" param:  {self.files['param']}")
        print(f" output: {self.files['data']}")
        print()

        state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
        self.display.update_image(state)
        cv2.waitKey(self.config['projector']['start_dt_ms'])

        self.find_bg_image()
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
        t_now = t_trial
        pos = None
        while t_now - t_trial < trial_param['duration']:
            t_now = time.time()
            t_elapsed = t_now - t_trial
            ok, image = self.camera.read()
            if ok:
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                diff_image = cv2.absdiff(gray_image, self.bg_image) 
                blob_list, blob_image, thresh_image = self.blob_finder.find(diff_image)

                if blob_list:
                    fly = get_max_area_blob(blob_list)
                    pos = (fly['centroid_x'], fly['centroid_y'])
            self.update_display(t_elapsed, pos, trial_param)

    def update_display(self, t, pos, trial_param): 
        display_mode = DisplayMode[trial_param['display_mode'].upper()] 
        if display_mode == DisplayMode.BLACK:
            kwargs = {}
        elif display_mode == DisplayMode.STATIC_IMAGE:
            kwargs = {'name': trial_param['name']} 
        elif display_mode == DisplayMode.ROTATING_RAYS:
            if (trial_param['center'] == 'arena') or (pos is None): 
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
        cv2.waitKey(1)

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

# ------------------------------------------------------------------------------------------------

def get_max_area_blob(blob_list):
    blob_area_array = np.array([blob['area'] for blob in blob_list])
    ind = blob_area_array.argmax()
    return blob_list[ind]


#class DisplayProcess:
#
#    def __init__(self, config, images=None):
#        self.done = False
#        self.data_queue = Queue()
#        self.done_event = Event()
#        self.task = DisplayTask(config, images, self.data_queue, self.done_event) 
#        self.process = Process(target=self.task.run,daemon=True)
#
#    def start(self):
#        self.process.start()
#
#    def stop(self):
#        self.process.stop()
#
#    def update_image(self,state):
#        self.data_queue.put(state)
#
#
#
#class DisplayTask:
#
#    def __init__(self, config, images, data_queue, done_event):
#        self.config = config
#        self.images = images
#        self.data_queue = data_queue
#        self.done_event = done_event
#
#    def set_display_black(self, wait_ms=1):
#        state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
#        self.display.update_image(state)
#        if wait_ms is not None:
#            cv2.waitKey(wait_ms)
#
#    def run(self):
#
#        self.display = DisplayController(self.config, images=self.images)
#        #state = {'mode': DisplayMode.BLACK, 'kwargs': {}}
#        #self.display.update_image(state)
#        #cv2.waitKey(1)
#
#        while not self.done_event.is_set():
#            try:
#                new_state = self.data_queue.get(False)
#            except queue.Empty:
#                new_state = {}
#            if new_state:
#                pass
#                #self.display.update_image(new_state)
#                #cv2.waitKey(1)


    



