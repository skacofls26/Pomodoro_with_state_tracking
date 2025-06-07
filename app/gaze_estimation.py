'''
gaze_estimation.py
    theta, phi 기반 집중 판단 로직

'''
import cv2
import numpy as np
import mediapipe as mp
import math
from config_model import OpenVINO_models

# ======================= 상수 정의 =======================

# 눈 이미지 추출을 위한 눈 좌표 인덱스
LEFT_EYE_LANDMARKS = (33, 133)
RIGHT_EYE_LANDMARKS = (362, 263)


# ======================= 보조 함수 정의 =======================

# FaceMesh 기반 눈 좌표 추출 함수
def denormalize_landmark(landmark, width, height):    
    return int(landmark.x * width), int(landmark.y * height)

# 눈 앞머리-뒷머리 중심점 기반 이미지 크롭 함수
def get_eye_crop(image, in_x, in_y, out_x, out_y, scale=1.8): 
    h, w = image.shape[:2]
    dist =  np.linalg.norm(np.array([in_x, in_y]) - np.array([out_x, out_y]))
    box_size = int(dist * scale)
    half = box_size // 2
    center_x, center_y = (in_x + out_x) // 2, (in_y + out_y) // 2
    x1, y1 = max(center_x - half, 0), max(center_y - half, 0) 
    x2, y2 = min(center_x + half, w), min(center_y + half, h) 
    return image[y1:y2, x1:x2]

# head pose의 Roll 기반 눈 이미지 정렬 함수
def rotate_image_around_center(image, angle_degrees):
    if image is None or image.size == 0:
        return None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# 시선 벡터 정규화 함수 
def normalize_gaze_vector(gaze_vector):
    norm = np.linalg.norm(gaze_vector)
    return gaze_vector if norm == 0 else gaze_vector / norm

# 전후 방향 머리 기울임 보정 함수
def apply_roll_alignment(gaze_vector, roll_degrees):   
    roll_radians = np.radians(roll_degrees)
    cs, sn = np.cos(roll_radians), np.sin(roll_radians)
    x, y, z = gaze_vector
    return np.array([x * cs + y * sn, -x * sn + y * cs, z])


# ======================= 집중 판단 함수 =======================

def is_focused(frame):
# ─── 0) 얼굴 추출 (OpenVINO의 "face" 모델) ───
    ih, iw = frame.shape[:2]
    resized = cv2.resize(frame, (672, 384))
    input_image = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    
# ─── 1) 얼굴 추출 (OpenVINO의 "face" 모델) ───
    face_model = OpenVINO_models["face"]
    face_output = face_model["model"]({face_model["input"]: input_image})
    detections = face_output[list(face_output.keys())[0]]

    for detection in detections[0][0]:  # 가장 신뢰도 높은 얼굴 처리 
        if detection[2] < 0.6:
            continue
        xmin = int(detection[3] * iw)
        ymin = int(detection[4] * ih)
        xmax = int(detection[5] * iw)
        ymax = int(detection[6] * ih)
        face = frame[ymin:ymax, xmin:xmax]
        if face.size == 0:
            continue
        
# ─── 2) Head Pose 추출 (OpenVINO의 "head_pose" 모델) ───
        headpose_input = cv2.resize(face, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        headpose_model = OpenVINO_models["head_pose"]
        headpose_results = headpose_model["model"]({headpose_model["input"]: headpose_input})
        yaw = headpose_results['angle_y_fc'][0][0]  # 머리의 좌우 방향 회전각 (°)
        pitch = headpose_results['angle_p_fc'][0][0]  # 머리의 상하 방향 회전각 (°)
        roll = headpose_results['angle_r_fc'][0][0]  # 머리의 전후 방향 회전각 (°)
        head_pose = np.array([[yaw, pitch, roll]], dtype=np.float32)  # Head Pose 추출 

# ─── 3) Landmarks 추출 (FaceMesh 모델) ───
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as FaceMesh:
            result = FaceMesh.process(face_rgb)
            if not result.multi_face_landmarks:
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            try:
                Lin_x, Lin_y = denormalize_landmark(landmarks[LEFT_EYE_LANDMARKS[0]], face.shape[1], face.shape[0])
                Lout_x, Lout_y = denormalize_landmark(landmarks[LEFT_EYE_LANDMARKS[1]], face.shape[1], face.shape[0])
                Rin_x, Rin_y = denormalize_landmark(landmarks[RIGHT_EYE_LANDMARKS[0]], face.shape[1], face.shape[0])
                Rout_x, Rout_y = denormalize_landmark(landmarks[RIGHT_EYE_LANDMARKS[1]], face.shape[1], face.shape[0])
            except:
                continue

    # ─── 4) 눈 이미지 추출 및 보정 (gaze 모델 입력 전처리) ───
            left_eye_patch = get_eye_crop(face, Lin_x, Lin_y, Lout_x, Lout_y)
            right_eye_patch = get_eye_crop(face, Rin_x, Rin_y, Rout_x, Rout_y)
            left_eye_patch = rotate_image_around_center(left_eye_patch, roll)
            right_eye_patch = rotate_image_around_center(right_eye_patch, roll)

            if left_eye_patch is None or right_eye_patch is None or left_eye_patch.size == 0 or right_eye_patch.size == 0:
                return None, None, False

            left_eye_image_input = cv2.resize(left_eye_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            right_eye_image_input = cv2.resize(right_eye_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


    # ─── 5) 시선 벡터 추정 및 보정 (OpenVINO의 "gaze" 모델) ───
            gaze_model = OpenVINO_models["gaze"]["model"]
            gaze_inputs = gaze_model.inputs
            input_names = [inp.get_any_name() for inp in gaze_inputs]
            gaze_res = gaze_model({
                input_names[0]: left_eye_image_input,
                input_names[1]: right_eye_image_input,
                input_names[2]: head_pose
            })
            # 시선 벡터 gaze_vector: 3차원 벡터 (gaze_x, gaze_y, gaze_z)
            gaze_vector = gaze_res[list(gaze_res.keys())[0]][0]  
            gaze_vector = normalize_gaze_vector(gaze_vector)
            gaze_vector = apply_roll_alignment(gaze_vector, roll)

            gaze_x = gaze_vector[0]  # 시선 벡터의 좌우 방향 투영값 (HeadPose의 Pitch)
            gaze_y = gaze_vector[1]  # 시선 벡터의 상하 방향 투영값 (HeadPose의 Yaw)
            gaze_z = gaze_vector[2]  # 시선 벡터의 전후 방향 투영값 (HeadPose의 Roll)

    # ─── 6) 시선 벡터 이용한 집중 판단 ───
            theta = math.atan2(gaze_x, abs(gaze_z))  # 시선 벡터로 계산한 수평 회전각 (rad)
            phi = math.atan2(gaze_y, abs(gaze_z))  # 시선 벡터로 계산한 수직 회전각 (rad)
            focused = abs(theta) < 0.40 and -0.50 < phi < 0.9
            return theta, phi, focused

    return None, None, False  # 얼굴 감지 실패