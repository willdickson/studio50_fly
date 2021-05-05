import cv2

class Camera(cv2.VideoCapture):

    def __init__(self, config, mode):
        param = config['camera']
        super().__init__(param['device'])
        if not self.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*param['fourcc'])):
            raise RuntimeError('unable to set fourcc to mjpg')
        if not self.set(cv2.CAP_PROP_FRAME_WIDTH, param['width']):
            raise RuntimeError('unable to set frame width')
        if not self.set(cv2.CAP_PROP_FRAME_HEIGHT, param['height']):
            raise RuntimeError('unable to set frame height')
        if not self.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
            raise RuntimeError('unable to set manual exposure')
        if mode == 'fly':
            exposure = config['fly']['exposure']
        elif mode == 'calibration-homography':
            exposure = config['calibration']['homography']['exposure']
        elif mode == 'calibration-position':
            exposure = config['calibration']['position']['exposure']
        else:
            raise ValueError(f'unknown mode {mode}')
        if not self.set(cv2.CAP_PROP_EXPOSURE, exposure):
            raise RuntimeError('unable to set exposure value')



