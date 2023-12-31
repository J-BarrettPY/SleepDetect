# SleepDetect

SleepDetect is a Python script for eye and gaze tracking using OpenCV and dlib. This script captures video from a webcam or a video file and detects whether the eyes are closed, if the person is sleeping, and the direction in which the person is looking.

## Prerequisites

You MUST download "Face Landmarks" release and place shape_predictor_68_face_landmarks.dat in the same directory!

Before running the SleepDetect script, make sure you have the following dependencies installed:
- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- dlib (`dlib`)
- argparse (`argparse`)

You can install these dependencies using `pip`:
```
pip install opencv-python numpy dlib argparse
```

## Usage
To use SleepDetect, run the script with the following command:

```
python SleepDetect.py --video <path_to_video_file>
```

Replace `<path_to_video_file>` with the path to the video file you want to analyze. If you don't specify a video file, the script will use the default webcam (camera index 0).

## Features
SleepDetect offers the following features:

- Detects faces using both Haar Cascade and dlib's face detector.
- Tracks eye landmarks and calculates the blinking ratio to determine if the eyes are closed.
- Displays "EYES CLOSED" and "SLEEPING" messages when the eyes are closed for a certain duration.
- Estimates the gaze direction of the person as "LOOKING RIGHT," "LOOKING CENTER," or "LOOKING LEFT."

## Controls
- Press the 'ESC' key to exit the application.

## License
This script is provided under the MIT License. You can find more details in the [LICENSE](LICENSE) file.
