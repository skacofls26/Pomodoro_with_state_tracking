import cv2
import mediapipe as mp

# ─────────────────── Mediapipe 초기화 ───────────────────
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────── 인덱스 정의 ───────────────────
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
LEFT_ELBOW    = mp_pose.PoseLandmark.LEFT_ELBOW.value
RIGHT_ELBOW   = mp_pose.PoseLandmark.RIGHT_ELBOW.value
LEFT_EAR      = mp_pose.PoseLandmark.LEFT_EAR.value
RIGHT_EAR     = mp_pose.PoseLandmark.RIGHT_EAR.value
LEFT_INDEX    = mp_pose.PoseLandmark.LEFT_INDEX.value
RIGHT_INDEX   = mp_pose.PoseLandmark.RIGHT_INDEX.value

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # FaceMesh
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ─────────────────── 웹캠 ───────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose.process(rgb)
    face_res = face.process(rgb)
    h, w = frame.shape[:2]

    # ── 1) Pose 랜드마크 시각화 ──
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        pose_points = {
            "L_Shoulder": lm[LEFT_SHOULDER],
            "R_Shoulder": lm[RIGHT_SHOULDER],
            "L_Elbow":    lm[LEFT_ELBOW],
            "R_Elbow":    lm[RIGHT_ELBOW],
            "L_Ear":      lm[LEFT_EAR],
            "R_Ear":      lm[RIGHT_EAR],
            "L_Index":    lm[LEFT_INDEX],
            "R_Index":    lm[RIGHT_INDEX],
        }

        for name, pt in pose_points.items():
            x, y = int(pt.x * w), int(pt.y * h)
            color = (0,255,0) if 'L_' in name else (0,0,255)  # 초록 L / 빨강 R
            cv2.circle(frame, (x, y), 6, color, -1)

    # ── 2) FaceMesh 눈 랜드마크 시각화 ──
    if face_res.multi_face_landmarks:
        fm = face_res.multi_face_landmarks[0].landmark
        for idx in LEFT_EYE_IDX:
            x, y = int(fm[idx].x * w), int(fm[idx].y * h)
            cv2.circle(frame, (x, y), 4, (255, 100,  50), -1)   # 주황
        for idx in RIGHT_EYE_IDX:
            x, y = int(fm[idx].x * w), int(fm[idx].y * h)
            cv2.circle(frame, (x, y), 4, ( 50, 100, 255), -1)   # 파랑

    # 안내 텍스트
    cv2.putText(frame, "Press 'q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imshow("Pose + Eye Landmarks Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
