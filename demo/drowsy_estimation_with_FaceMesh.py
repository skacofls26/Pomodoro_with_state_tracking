import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time

def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 눈 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 임계값
EAR_THRESH = 0.21
BLINK_MAX_DURATION = 0.3  # blink 최대 지속 시간 (초)
DROWSY_MIN_DURATION = 1.0  # 졸음 최소 지속 시간 (초)

blink_count = 0
drowsy = False
eye_closed_start = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        mesh = result.multi_face_landmarks[0].landmark
        left_eye = np.array([(mesh[i].x * w, mesh[i].y * h) for i in LEFT_EYE_IDX])
        right_eye = np.array([(mesh[i].x * w, mesh[i].y * h) for i in RIGHT_EYE_IDX])

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        now = time.time()

        if ear < EAR_THRESH:
            if eye_closed_start is None:
                eye_closed_start = now
            else:
                duration = now - eye_closed_start
                if duration >= DROWSY_MIN_DURATION:
                    drowsy = True
        else:
            if eye_closed_start is not None:
                duration = now - eye_closed_start
                if duration <= BLINK_MAX_DURATION:
                    blink_count += 1
                eye_closed_start = None
                drowsy = False  # 눈 떴으므로 졸음 해제

        # 눈 시각화
        for (x, y) in np.concatenate((left_eye, right_eye)):
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        # 텍스트 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if drowsy:
            cv2.putText(frame, "DROWSY!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        eye_closed_start = None
        drowsy = False
        cv2.putText(frame, "Detecting face...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("Blink & Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
