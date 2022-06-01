import cv2
import numpy as np

class PoseVisualizer:
    def __init__(self, axis_length=3):
        self.axis = np.float32([
            [axis_length, 0, 0], 
            [0, axis_length, 0], 
            [0, 0, -axis_length]
        ]).reshape(-1, 3)
    
    def draw_axes(self, img, corners, imgpts):
        """Draw 3D coordinate axes on image"""
        corner = tuple(corners[0].ravel().astype(int))
        img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
        return img
    
    def project_points(self, points, rvec, tvec, camera_matrix, dist_coeffs):
        """Project 3D points to 2D image plane"""
        imgpts, jac = cv2.projectPoints(
            points, rvec, tvec, camera_matrix, dist_coeffs
        )
        return imgpts
    
    def draw_pose(self, img, corners, rvec, tvec, camera_matrix, dist_coeffs):
        """Draw complete pose visualization on image"""
        imgpts = self.project_points(
            self.axis, rvec, tvec, camera_matrix, dist_coeffs
        )
        return self.draw_axes(img, corners, imgpts)