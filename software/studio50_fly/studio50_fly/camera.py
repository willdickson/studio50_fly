import cv2

class Camera(cv2.VideoCapture):

    def __init__(self,param):
        super().__init__(param['device'])
        if not self.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*param['fourcc'])):
            raise RuntimeError('unable to set fourcc to mjpg')
        if not self.set(cv2.CAP_PROP_FRAME_WIDTH, param['width']):
            raise RuntimeError('unable to set frame width')
        if not self.set(cv2.CAP_PROP_FRAME_HEIGHT, param['height']):
            raise RuntimeError('unable to set frame height')
        if not self.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3):
            raise RuntimeError('unable to set auto exposure')



