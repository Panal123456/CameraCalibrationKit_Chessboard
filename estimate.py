import argparse
import cv2
import os
from glob import glob
import numpy as np
from utils.camera import CameraCalibrator
from utils.chessboard import ChessboardDetector
from utils.visualization import PoseVisualizer

def main():
    parser = argparse.ArgumentParser(description='Pose estimation from chessboard images')
    parser.add_argument('--pattern', type=int, nargs=2, default=[7, 9],
                        help='Chessboard pattern size (default: 7x9)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output images')
    parser.add_argument('--calibration', type=str, default='config/calibration_params.npy',
                        help='Camera calibration file')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale factor for output images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load calibration parameters
    calibrator = CameraCalibrator()
    mtx, dist = calibrator.load_calibration(args.calibration)
    
    # Create detector and visualizer
    detector = ChessboardDetector(pattern_size=tuple(args.pattern))
    visualizer = PoseVisualizer()
    
    # Process images
    image_paths = glob(os.path.join(args.image_dir, '*.png')) + \
                  glob(os.path.join(args.image_dir, '*.jpg'))
    
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        img = cv2.imread(image_path)
        
        # Detect chessboard
        ret, corners = detector.detect(img)
        
        if ret:
            # Estimate pose
            ret, rvec, tvec, inliers = detector.estimate_pose(corners, mtx, dist)
            
            if ret:
                # Draw pose visualization
                img = visualizer.draw_pose(img, corners, rvec, tvec, mtx, dist)
        
        # Resize and save image
        if args.scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"pose_{filename}")
        cv2.imwrite(output_path, img)
        print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()