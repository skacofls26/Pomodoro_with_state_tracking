import cv2
import mediapipe as mp
import math
import numpy as np
import time
from scipy.spatial import distance as dist

# ─────────────────── Mediapipe 초기화 ───────────────────
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────── 사용할 랜드마크 인덱스 ───────────────────
LEFT_EAR_IDX   = 234   # 왼쪽 귀(Tragion)
RIGHT_EAR_IDX  = 454   # 오른쪽 귀(Tragion)
NOSE_TIP_IDX   =   1   # 코끝

LEFT_EYE_IDX   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
EAR_THRESH     = 0.21
DROWSY_THRESH  = 4.0

def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def is_drowsy(mesh_list, h, w):
    # 얼굴 랜드마크가 없으면 EAR 계산 불가
    if not mesh_list:
        return None, False

    mesh = mesh_list[0].landmark
    left_eye  = np.array([(mesh[i].x * w, mesh[i].y * h) for i in LEFT_EYE_IDX])
    right_eye = np.array([(mesh[i].x * w, mesh[i].y * h) for i in RIGHT_EYE_IDX])
    ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
    now = time.monotonic()

    # 상태 초기화
    if not hasattr(is_drowsy, "drowsy"):
        is_drowsy.drowsy = False
        is_drowsy.eye_closed_start = None

    if ear < EAR_THRESH:
        if is_drowsy.eye_closed_start is None:
            is_drowsy.eye_closed_start = now
        elif not is_drowsy.drowsy and (now - is_drowsy.eye_closed_start) >= DROWSY_THRESH:
            is_drowsy.drowsy = True
    else:
        is_drowsy.eye_closed_start = None
        is_drowsy.drowsy = False

    return ear, is_drowsy.drowsy


# ─────────────────── 웹캠 설정 ───────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_res = face.process(rgb)

    # ─── 1) 항상 시각화: 귀와 코끝 좌표 ───
    if face_res.multi_face_landmarks:
        mesh = face_res.multi_face_landmarks[0].landmark

        lx, ly = int(mesh[LEFT_EAR_IDX].x * w), int(mesh[LEFT_EAR_IDX].y * h)
        rx, ry = int(mesh[RIGHT_EAR_IDX].x * w), int(mesh[RIGHT_EAR_IDX].y * h)
        nx, ny = int(mesh[NOSE_TIP_IDX].x * w), int(mesh[NOSE_TIP_IDX].y * h)

        # 점 찍기
        cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)   # 파랑: 왼쪽 귀
        cv2.circle(frame, (rx, ry), 5, (0, 255, 0), -1)   # 초록: 오른쪽 귀
        cv2.circle(frame, (nx, ny), 5, (0, 255, 255), -1) # 노랑: 코끝

        # 좌표 텍스트
        base_y = h - 60
        cv2.putText(frame, f"Left Ear:  ({lx},{ly})",   (10, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Right Ear: ({rx},{ry})",   (10, base_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Nose Tip:  ({nx},{ny})",   (10, base_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ─── 2) 졸음 판단 & EAR 감지 실패 표시 ───
    ear, drowsy = is_drowsy(face_res.multi_face_landmarks, h, w)
    if ear is None:
        cv2.putText(frame, "No detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ─── 3) 종료 안내 & 화면 출력 ───
    cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imshow("Face + Drowsiness + Keypoints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
