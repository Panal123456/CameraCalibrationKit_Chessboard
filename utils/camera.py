import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self, pattern_size=(7, 9)):
        self.pattern_size = pattern_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        
    def find_chessboard_corners(self, image_paths):
        """Find chessboard corners in a set of images"""
        for path in image_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            
            if ret:
                # Refine corner locations
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                
        return len(self.objpoints) > 0
    
    def calibrate(self, image_size):
        """Calibrate camera using collected points"""
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        return ret, mtx, dist, rvecs, tvecs
    
    def save_calibration(self, filename, mtx, dist):
        """Save calibration parameters to file"""
        np.save(filename, {'camera_matrix': mtx, 'dist_coeffs': dist})
    
    def load_calibration(self, filename):
        """Load calibration parameters from file"""
        data = np.load(filename, allow_pickle=True).item()
        return data['camera_matrix'], data['dist_coeffs']
