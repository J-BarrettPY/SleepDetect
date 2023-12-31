import cv2
import numpy as np
import dlib
import time
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Eye and Gaze Tracking')
parser.add_argument('--video', type=str, help='Path to video file', default=None)
args = parser.parse_args()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to calculate the ratio of horizontal and vertical eye landmarks
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Calculate the horizontal and vertical distances
    hor_line_length = np.linalg.norm(np.array(left_point) - np.array(right_point))
    ver_line_length = np.linalg.norm(np.array(center_top) - np.array(center_bottom))

    ratio = hor_line_length / ver_line_length
    return ratio

# Function to get gaze direction
def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) for i in range(len(eye_points))], np.int32)
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Start the webcam capture
cap = cv2.VideoCapture(args.video if args.video else 0)

eyes_closed_time = None
sleeping_displayed = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # First use Haar Cascade to detect faces
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Then use dlib's detector
    faces_dlib = detector(gray)

    for (x, y, w, h) in faces_haar:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for face in faces_dlib:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 4.5:
            # Check if "SLEEPING" has not been displayed before showing "EYES CLOSED"
            if not sleeping_displayed:
                cv2.putText(frame, "EYES CLOSED", (50, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 170, 255), 2)

            # Check if eyes just closed
            if eyes_closed_time is None:
                eyes_closed_time = time.time()
            elif time.time() - eyes_closed_time > 1:
                cv2.putText(frame, "SLEEPING", (50, 140), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                sleeping_displayed = True  # Set the flag once "SLEEPING" is displayed
        else:
            eyes_closed_time = None
            sleeping_displayed = False



        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        if gaze_ratio <= 1:
            cv2.putText(frame, "LOOKING RIGHT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        elif 1 < gaze_ratio < 1.7:
            cv2.putText(frame, "LOOKING CENTER", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "LOOKING LEFT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Eye Tracking", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
