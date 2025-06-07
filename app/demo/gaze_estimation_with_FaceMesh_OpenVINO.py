import cv2
import numpy as np
import time
import math
from openvino import Core
import mediapipe as mp

# OpenVINO 모델 경로 (시선 추정용)
model_dir = "../OpenVINO"

# FaceMesh 모델 로드 (좌표 추출용)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# 눈 이미지 추출을 위한 눈 좌표 인덱스 
LEFT_EYE_LANDMARKS = (33, 133)
RIGHT_EYE_LANDMARKS = (362, 263)

def load_OpenVINO_model(core, model_path, device="CPU"):
    model = core.read_model(model_path + ".xml")
    compiled = core.compile_model(model, device)
    return compiled

def denormalize_landmark(landmark, width, height):
    # FaceMesh 기반 눈 좌표 추출 함수
    return int(landmark.x * width), int(landmark.y * height)

def get_eye_crop(image, in_x, in_y, out_x, out_y, scale=1.8): 
    # 눈 앞머리와 뒷머리의 중심점 기반 이미지 크롭 함수
    h, w = image.shape[:2]
    dist =  np.linalg.norm(np.array([in_x, in_y]) - np.array([out_x, out_y]))
    box_size = int(dist * scale)
    half = box_size // 2
    center_x, center_y = (in_x + out_x) // 2, (in_y + out_y) // 2
    x1, y1 = max(center_x - half, 0), max(center_y - half, 0) 
    x2, y2 = min(center_x + half, w), min(center_y + half, h) 
    return image[y1:y2, x1:x2], center_x, center_y

def rotate_image_around_center(image, angle_degrees):
    # head pose의 Roll 값 기반 눈 이미지 정렬 함수
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def normalize_gaze_vector(gaze_vector):
    # 시선 벡터 정규화 함수 
    norm = np.linalg.norm(gaze_vector)
    return gaze_vector if norm == 0 else gaze_vector / norm

def apply_roll_alignment(gaze_vector, roll_degrees):
    # 고개 기울임 보정 함수 
    roll_radians = np.radians(roll_degrees)
    cs, sn = np.cos(roll_radians), np.sin(roll_radians)
    x, y, z = gaze_vector
    return np.array([x * cs + y * sn, -x * sn + y * cs, z])

def draw_gaze_vector(img, origin, gaze_vector, scale=100, color=(0, 255, 0)):
    # 시선 벡터 시각화 함수 
    x, y = origin
    dx = int(gaze_vector[0] * scale)
    dy = int(-gaze_vector[1] * scale)  # 좌우 반전
    cv2.arrowedLine(img, (x, y), (x + dx, y + dy), color, 2, tipLength=0.3)

def draw_eye_box_on_frame(frame, face_origin, center_x, center_y, box_size, color):
    xmin, ymin = face_origin
    half = box_size // 2
    x1, y1 = xmin + center_x - half, ymin + center_y - half
    x2, y2 = xmin + center_x + half, ymin + center_y + half
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def main():
    core = Core()
    models = {
        "face": load_OpenVINO_model(core, f"{model_dir}/face-detection-adas-0001/FP32/face-detection-adas-0001"),
        "head_pose": load_OpenVINO_model(core, f"{model_dir}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"),
        "gaze": load_OpenVINO_model(core, f"{model_dir}/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"),
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ih, iw = frame.shape[:2]
        resized = cv2.resize(frame, (672, 384))
        input_image = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        face_output = models["face"]({0: input_image})
        detections = face_output[list(face_output.keys())[0]]

        for det in detections[0][0]:
            if det[2] < 0.6:
                continue
            xmin = int(det[3] * iw)
            ymin = int(det[4] * ih)
            xmax = int(det[5] * iw)
            ymax = int(det[6] * ih)
            face = frame[ymin:ymax, xmin:xmax]
            if face.size == 0:
                continue

            # Head Pose 추출
            hp_input = cv2.resize(face, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            hp_res = models["head_pose"]({0: hp_input})
            yaw = hp_res['angle_y_fc'][0][0]
            pitch = hp_res['angle_p_fc'][0][0]
            roll = hp_res['angle_r_fc'][0][0]
            head_pose = np.array([[yaw, pitch, roll]], dtype=np.float32)

            # Landmarks 추출
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(face_rgb)
            if not result.multi_face_landmarks:
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            Lin_x, Lin_y = denormalize_landmark(landmarks[LEFT_EYE_LANDMARKS[0]], face.shape[1], face.shape[0])
            Lout_x, Lout_y = denormalize_landmark(landmarks[LEFT_EYE_LANDMARKS[1]], face.shape[1], face.shape[0])
            Rin_x, Rin_y = denormalize_landmark(landmarks[RIGHT_EYE_LANDMARKS[0]], face.shape[1], face.shape[0])
            Rout_x, Rout_y = denormalize_landmark(landmarks[RIGHT_EYE_LANDMARKS[1]], face.shape[1], face.shape[0])
            if not (0 <= Lin_x < face.shape[1] and 0 <= Lin_y < face.shape[0]):
                continue

            # 눈 이미지 추출 및 보정
            left_patch, Lcenter_x, Lcenter_y = get_eye_crop(face, Lin_x, Lin_y, Lout_x, Lout_y)
            right_patch, Rcenter_x, Rcenter_y = get_eye_crop(face, Rin_x, Rin_y, Rout_x, Rout_y) 
            if left_patch is None or left_patch.size == 0:
                return None, Lcenter_x, Lcenter_y
            if right_patch is None or right_patch.size == 0:
                return None, Rcenter_x, Rcenter_y
            
            left_patch = rotate_image_around_center(left_patch, roll)
            right_patch = rotate_image_around_center(right_patch, roll)
            
            # 눈 좌표 시각화 (눈 앞머리/뒷머리)
            for (x, y) in [(Lin_x, Lin_y), (Lout_x, Lout_y), (Rin_x, Rin_y), (Rout_x, Rout_y)]:
                cv2.circle(face, (x, y), 2, (255, 0, 0), -1)  # 파란 점

            # 눈 박스 시각화 (좌표는 얼굴 crop 기준이므로 전체 프레임 좌표로 보정 필요)
            def draw_eye_box_on_frame(frame, face_origin, center_x, center_y, box_size, color):
                xmin, ymin = face_origin
                half = box_size // 2
                x1, y1 = xmin + center_x - half, ymin + center_y - half
                x2, y2 = xmin + center_x + half, ymin + center_y + half
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # box size 계산 재사용
            L_dist = np.linalg.norm(np.array([Lin_x, Lin_y]) - np.array([Lout_x, Lout_y]))
            R_dist = np.linalg.norm(np.array([Rin_x, Rin_y]) - np.array([Rout_x, Rout_y]))
            L_box = int(L_dist * 1.8)
            R_box = int(R_dist * 1.8)

            # 눈 박스 프레임에 시각화
            draw_eye_box_on_frame(frame, (xmin, ymin), Lcenter_x, Lcenter_y, L_box, (255, 255, 0))  # 왼쪽 노란 박스
            draw_eye_box_on_frame(frame, (xmin, ymin), Rcenter_x, Rcenter_y, R_box, (255, 255, 0))  # 오른쪽 노란 박스

            if left_patch.size == 0 or right_patch.size == 0:
                continue

            left_input = cv2.resize(left_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            right_input = cv2.resize(right_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # 시선 벡터 추정 및 보정 
            gaze_res = models["gaze"]({
                'left_eye_image': left_input,
                'right_eye_image': right_input,
                'head_pose_angles': head_pose
            })
            gaze_vector = gaze_res[list(gaze_res.keys())[0]][0]  # 시선 벡터 gaze_vector는 3차원 벡터 (gaze_x, gaze_y, gaze_z)
            gaze_vector = normalize_gaze_vector(gaze_vector)
            gaze_vector = apply_roll_alignment(gaze_vector, head_pose[0][2])
            gaze_x = gaze_vector[0]  # 시선 벡터의 좌우 방향 투영값
            gaze_y = gaze_vector[1]  # 시선 벡터의 상하 방향 투영값
            gaze_z = gaze_vector[2]  # 시선 벡터의 전후 방향 투영값

            # 시선 벡터 이용한 집중 여부 판단 
            theta = math.atan2(gaze_x, abs(gaze_z))  # 시선 벡터로 계산한 수평 회전각 (라디안 단위)
            phi = math.atan2(gaze_y, abs(gaze_z))    # 시선 벡터로 계산한 수직 회전각 (라디안 단위)
            focused = abs(theta) < 0.37 and -0.37 < phi < 0.12
            status = "Focused" if focused else "Not Focused"

            # 4개 좌표 평균점을 원점으로 시선 벡터 시각화
            cx = xmin + (Lcenter_x + Rcenter_x) // 2
            cy = ymin + (Lcenter_y + Rcenter_y) // 2
            draw_gaze_vector(frame, (cx, cy), gaze_vector, scale=80)

            cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if focused else (0, 0, 255), 2)

            # FPS 기록 
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                print(f"FPS: {fps:.1f}")
                prev_time = current_time
                frame_count = 0

            print(f"θ: {theta:.2f}, φ: {phi:.2f}, Roll: {roll:.2f} → {status}")
            break

        cv2.imshow("Gaze Visualization", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

main()