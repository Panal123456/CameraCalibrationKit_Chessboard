import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from utils.camera import CameraCalibrator
from utils.chessboard import ChessboardDetector
from utils.visualization import PoseVisualizer

class RealTimePoseEstimator:
    def __init__(self, pattern_size, calibration_file):
        # Load calibration parameters
        calibrator = CameraCalibrator()
        self.mtx, self.dist = calibrator.load_calibration(calibration_file)
        
        # Create detector and visualizer
        self.detector = ChessboardDetector(pattern_size=pattern_size)
        self.visualizer = PoseVisualizer()
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Video writer
        self.writer = None
        
    def start(self, record=False):
        """Start the RealSense pipeline"""
        self.pipeline.start(self.config)
        
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
    
    def process_frame(self, frame):
        """Process a single frame for pose estimation"""
        # Detect chessboard
        ret, corners = self.detector.detect(frame)
        
        if ret:
            # Estimate pose
            ret, rvec, tvec, inliers = self.detector.estimate_pose(corners, self.mtx, self.dist)
            
            if ret:
                # Draw pose visualization
                frame = self.visualizer.draw_pose(frame, corners, rvec, tvec, self.mtx, self.dist)
        
        return frame
    
    def run(self, record=False, save_dir=None):
        """Run the real-time pose estimation loop"""
        self.start(record=record)
        frame_count = 0
        
        try:
            while True:
                # Wait for a coherent frame
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write to video if recording
                if record and self.writer:
                    self.writer.write(processed_frame)
                
                # Display the resulting frame
                cv2.imshow('Real-Time Pose Estimation', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') and save_dir:  # Save frame
                    frame_count += 1
                    frame_filename = f"{save_dir}/frame_{frame_count:04d}.jpg"
                    cv2.imwrite(frame_filename, processed_frame)
                    print(f"Saved frame: {frame_filename}")
                elif key == ord('q'):  # Quit
                    break
                    
        finally:
            self.stop()
    
    def stop(self):
        """Stop the pipeline and release resources"""
        self.pipeline.stop()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Real-time pose estimation from chessboard')
    parser.add_argument('--pattern', type=int, nargs=2, default=[7, 9],
                        help='Chessboard pattern size (default: 7x9)')
    parser.add_argument('--calibration', type=str, default='config/calibration_params.npy',
                        help='Camera calibration file')
    parser.add_argument('--record', action='store_true',
                        help='Record video output')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save captured frames')
    
    args = parser.parse_args()
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create and run estimator
    estimator = RealTimePoseEstimator(
        pattern_size=tuple(args.pattern),
        calibration_file=args.calibration
    )
    
    estimator.run(record=args.record, save_dir=args.save_dir)

if __name__ == "__main__":
    main()