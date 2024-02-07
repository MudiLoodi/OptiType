import cv2
import dlib

# Load Dlib's pre-trained face and landmark detector
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_cap = cv2.VideoCapture(0)

calibration_points = [(100, 100), (300, 100), (500, 100)]  # Example calibration points
current_calibration_point = 0
calibration_data = []

def detect_pupil(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)

    for face in faces:
        landmarks = landmark_predictor(gray_image, face)
        x = landmarks.part(36).x
        y = landmarks.part(36).y
        # Calculate gaze point as the midpoint of the pupils
        circle = cv2.circle(vid, (x, y), 3, (0, 0, 255), 2)
        return circle

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
