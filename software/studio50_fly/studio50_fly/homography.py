import os
import cv2
import pickle
import numpy as np


class Homography:

    def __init__(self, config, cal_data=None):
        self.config = config
        if cal_data is not None:
            self.cal_data = {
                    'camera_pts'    : np.array(cal_data['camera_pts']),
                    'projector_pts' : np.array(cal_data['projector_pts']),
                    }
        else:
            self.load_calibration()
        self.find_matrix()

    @property
    def calibration_file(self):
        filename = self.config['calibration']['homography']['filename']
        return os.path.join(self.config.path, filename)

    def load_calibration(self): 
        with open(self.calibration_file,'rb') as f:
            self.cal_data = pickle.load(f)

    def save_calibration(self):
        with open(self.calibration_file,'wb') as f:
            pickle.dump(self.cal_data,f)

    def find_matrix(self):
        camera_pts = self.cal_data['camera_pts']
        projector_pts = self.cal_data['projector_pts']
        homography_matrix, _ = cv2.findHomography(camera_pts, projector_pts, cv2.RANSAC)
        self.matrix = homography_matrix.transpose()

    def camera_to_projector(self,camera_pts):
        _camera_pts = np.array(camera_pts)
        ndim = _camera_pts.ndim
        if ndim == 1:
            _camera_pts = np.reshape(_camera_pts,(1,2))
        camera_pts_3d = np.ones((_camera_pts.shape[0],3))
        camera_pts_3d[:,:2] = _camera_pts
        projector_pts_3d = np.dot(camera_pts_3d, self.matrix)
        denom = np.zeros((_camera_pts.shape[0],2))
        denom[:,0] = projector_pts_3d[:,2]
        denom[:,1] = projector_pts_3d[:,2]
        projector_pts = projector_pts_3d[:,:2]/denom
        if ndim == 1:
            projector_pts = np.reshape(projector_pts,(2,))
        return projector_pts

    def projector_to_camera(slef, projector_pts):
        _projector_pts = np.array(projector_pts)
        ndim = _projector_pts.ndim
        if ndim == 1:
            _projector_pts = np.reshape(_projector_pts, (1,2))
        projector_pts_3d = np.ones((_projector_pts.shape[0],3))
        projector_pts_3d[:,:2] = _projector_pts
        camera_pts_3d = np.dot(projector_pts_3d, np.linalg.inv(self.matrix))
        denom = np.zeros((camera_pts.shape[0],2))
        denom[:,0] = camera_pts_3d[:,2]
        denom[:,1] = camera_pts_3d[:,2]
        camera_pts = camera_pts_3d[:,:2]/denom
        if ndim == 1:
            camera_pts = np.reshape(camera_pts,(2,))
        return camera_pts

    def reproject_cal_data(self):
        return self.camera_to_projector(self.cal_data['camera_pts'])

    def reprojection_error(self):
        projector_pts_pred = self.reproject_cal_data()
        error = (self.cal_data['projector_pts'] - projector_pts_pred)**2
        error = error.sum(axis=1)
        error = np.sqrt(error)
        error = error.mean()
        return error

    def calibration_data(self,jsonable=True):
        if jsonable:
            cal_data = {
                    'camera_pts'    : self.cal_data['camera_pts'].tolist(),
                    'projector_pts' : self.cal_data['projector_pts'].tolist(),
                    }
            return cal_data
        else:
            return self.cal_data


