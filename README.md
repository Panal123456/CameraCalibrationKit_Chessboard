# Chessboard Pose Estimation

A modular system for camera calibration and chessboard pose estimation using OpenCV.

## Features

- Camera calibration using chessboard pattern
- Batch processing of images for pose estimation
- Real-time pose estimation using Intel RealSense cameras
- Visualization of 3D coordinate axes on detected chessboards

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Camera Calibration

Place calibration images in a directory and run:
```
python calibrate.py --image_dir path/to/calibration/images --output config/calibration_params.npy
```

### 2. Batch Pose Estimation

Process a directory of images:
```
python estimate.py --image_dir path/to/input/images --output_dir path/to/output/directory
```

### 3. Real-time Pose Estimation

Run real-time pose estimation with a RealSense camera:
```
python realtime.py --record --save_dir path/to/save/frames
```

Press 's' to save a frame, 'q' to quit.

## Project Structure

- `calibrate.py`: Camera calibration script
- `estimate.py`: Batch image processing script
- `realtime.py`: Real-time pose estimation script
- `utils/`: Modular utility classes
  - `camera.py`: Camera calibration utilities
  - `chessboard.py`: Chessboard detection and pose estimation
  - `visualization.py`: Visualization utilities
- `config/`: Configuration directory for calibration parameters
