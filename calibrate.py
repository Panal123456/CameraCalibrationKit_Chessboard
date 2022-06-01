import argparse
import cv2
import os
from glob import glob
from utils.camera import CameraCalibrator

def main():
    parser = argparse.ArgumentParser(description='Camera calibration using chessboard pattern')
    parser.add_argument('--pattern', type=int, nargs=2, default=[7, 9],
                        help='Chessboard pattern size (default: 7x9)')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing calibration images')
    parser.add_argument('--output', type=str, default='config/calibration_params.npy',
                        help='Output file for calibration parameters')
    
    args = parser.parse_args()
    
    # Get all image paths
    image_paths = glob(os.path.join(args.image_dir, '*.png')) + \
                  glob(os.path.join(args.image_dir, '*.jpg'))
    
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    # Create calibrator
    calibrator = CameraCalibrator(pattern_size=tuple(args.pattern))
    
    # Find chessboard corners
    print("Finding chessboard corners...")
    success = calibrator.find_chessboard_corners(image_paths)
    
    if not success:
        print("Failed to find chessboard corners in any image")
        return
    
    # Calibrate camera
    print("Calibrating camera...")
    sample_image = cv2.imread(image_paths[0])
    image_size = (sample_image.shape[1], sample_image.shape[0])
    
    ret, mtx, dist, rvecs, tvecs = calibrator.calibrate(image_size)
    
    if ret:
        # Save calibration parameters
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        calibrator.save_calibration(args.output, mtx, dist)
        print(f"Calibration successful! Parameters saved to {args.output}")
        
        # Print calibration results
        print("Camera matrix:")
        print(mtx)
        print("\nDistortion coefficients:")
        print(dist)
    else:
        print("Calibration failed")

if __name__ == "__main__":
    main()