import cv2
import dlib
from math import hypot

# Load Dlib's pre-trained face and landmark detector
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_cap = cv2.VideoCapture(0)

calibration_points = [(100, 100), (300, 100), (500, 100)]  # Example calibration points
current_calibration_point = 0
calibration_data = []

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def detect_pupil(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)

    for face in faces:
        landmarks = landmark_predictor(gray_image, face)
        
        left_eye_ratio = detect_blink(vid, landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_ratio = detect_blink(vid, landmarks, [42, 43, 44, 45, 46, 47])
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        if blinking_ratio > 5.6:
            cv2.putText(vid, "Blinking", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))

def detect_blink(vid, eye_landmarks, eye_points):
    left_point = (eye_landmarks.part(eye_points[0]).x, eye_landmarks.part(eye_points[0]).y)
    right_point = (eye_landmarks.part(eye_points[3]).x, eye_landmarks.part(eye_points[3]).y)
    center_top = midpoint(eye_landmarks.part(eye_points[1]), eye_landmarks.part(eye_points[2]))
    center_bottom = midpoint(eye_landmarks.part(eye_points[5]), eye_landmarks.part(eye_points[4]))
    # circle = cv2.circle(vid, (center_top), 3, (0, 0, 255), 2)
    
    hor_line  = cv2.line(vid, left_point, right_point, (0,255,0), 2)
    ver_line = cv2.line(vid, center_top, center_bottom, (0,255,0), 2)

    hor_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    ratio = hor_line_length/ver_line_length
    return ratio

# Main loop for eye tracking using calibration mapping
while True:
    result, video_frame = video_cap.read()
    if result is False:
        print("Failed to read frame")
        break

    gaze_point = detect_pupil(video_frame)
    

    # Draw gaze point on the frame
    cv2.imshow("Eye Tracking", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()
