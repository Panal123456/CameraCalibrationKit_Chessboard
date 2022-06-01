import numpy as np
import cv2

class ChessboardDetector:
    def __init__(self, pattern_size=(7, 9)):
        self.pattern_size = pattern_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
    def detect(self, image, refine=True):
        """Detect chessboard in image and return corners if found"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret and refine:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
        return ret, corners if ret else None
    
    def estimate_pose(self, corners, camera_matrix, dist_coeffs):
        """Estimate pose of chessboard relative to camera"""
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.objp, corners, camera_matrix, dist_coeffs
        )
        return ret, rvec, tvec, inliers