import cv2
import numpy as np
import time
import math
from openvino.runtime import Core

model_dir = "./open_model_zoo/demos/gaze_estimation_demo/cpp/intel"

def load_model(core, model_path, device="CPU"):
    model = core.read_model(model_path + ".xml")
    compiled = core.compile_model(model, device)
    input_tensor = compiled.input(0)
    return compiled, input_tensor

def crop_eye_patch(img, x, y, size):
    h, w = img.shape[:2]
    half = size // 2
    if x - half < 0 or y - half < 0 or x + half > w or y + half > h:
        return None
    return img[y - half:y + half, x - half:x + half]

def draw_gaze_vector(img, origin, gaze_vector, scale=100, color=(0, 255, 0)):
    x, y = map(int, origin)  # origin을 확실히 int로
    dx = int(gaze_vector[0] * scale)
    dy = int(-gaze_vector[1] * scale)
    try:
        cv2.arrowedLine(img, (x, y), (x + dx, y + dy), color, 2, tipLength=0.3)
    except Exception as e:
        print(f"⚠️ arrowedLine 예외 발생: {e} | origin=({x}, {y}), dx={dx}, dy={dy}")

def main():
    core = Core()
    models = {
        "face": load_model(core, f"{model_dir}/face-detection-adas-0001/FP32/face-detection-adas-0001")[0],
        "head_pose": load_model(core, f"{model_dir}/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001")[0],
        "landmarks": load_model(core, f"{model_dir}/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002")[0],
        "gaze": load_model(core, f"{model_dir}/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002")[0],
    }

    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():
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

            # Head Pose
            hp_input = cv2.resize(face, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            hp_res = models["head_pose"]({0: hp_input})
            hp_keys = list(hp_res.keys())
            head_pose = np.array([[hp_res[hp_keys[0]][0][0], hp_res[hp_keys[1]][0][0], hp_res[hp_keys[2]][0][0]]], dtype=np.float32)

            # Landmarks
            lm_input = cv2.resize(face, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            lm_res = models["landmarks"]({0: lm_input})
            landmarks = lm_res[list(lm_res.keys())[0]][0]

            # 디버깅용 출력 및 시각화
            print(f"[Landmarks 0~5]: {[round(landmarks[i], 2) for i in range(6)]}")

            for i in range(0, 70, 2):
                x = int(landmarks[i] * face.shape[1])
                y = int(landmarks[i + 1] * face.shape[0])
                cv2.circle(face, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(face, str(i//2), (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            # 일단 기존 인덱스로 진행
            lx, ly = int(landmarks[0] * face.shape[1]), int(landmarks[1] * face.shape[0])
            rx, ry = int(landmarks[4] * face.shape[1]), int(landmarks[3] * face.shape[0])
            print(f"왼쪽 눈 좌표: ({lx},{ly}), 오른쪽 눈 좌표: ({rx},{ry})")

            left_patch = crop_eye_patch(face, lx, ly, 60)
            right_patch = crop_eye_patch(face, rx, ry, 60)

            if left_patch is None:
                print("⚠️ 왼쪽 눈 crop 실패")
            if right_patch is None:
                print("⚠️ 오른쪽 눈 crop 실패")

            inputs = {}
            eye_count = 0
            if left_patch is not None:
                left_input = cv2.resize(left_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                inputs['left_eye_image'] = left_input
                eye_count += 1
            if right_patch is not None:
                right_input = cv2.resize(right_patch, (60, 60)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                inputs['right_eye_image'] = right_input
                eye_count += 1
            if eye_count == 0:
                continue

            inputs['head_pose_angles'] = head_pose
            gaze_res = models["gaze"](inputs)
            gaze_vector = gaze_res[list(gaze_res.keys())[0]][0]

            gaze_z = gaze_vector[2]
            theta = math.atan2(gaze_vector[0], gaze_vector[2])
            phi = math.atan2(gaze_vector[1], gaze_vector[2])
            focused = abs(theta) < 0.35 and abs(phi) < 0.35
            status = "Focused" if focused else "Not Focused"

            # draw 전에 모든 좌표를 int로 강제
            if left_patch is not None and right_patch is not None:
                center_x = int(xmin + (lx + rx) // 2)
                center_y = int(ymin + (ly + ry) // 2)
            elif left_patch is not None:
                center_x = int(xmin + lx)
                center_y = int(ymin + ly)
            elif right_patch is not None:
                center_x = int(xmin + rx)
                center_y = int(ymin + ry)
            else:
                print("⚠️ 두 눈 patch 모두 실패 → draw 생략")
                continue

            # gaze_vector 자체가 문제가 없는지도 검사
            if any(map(lambda v: not np.isfinite(v), gaze_vector)):
                print("⚠️ gaze_vector에 NaN 또는 Inf 포함 → 생략")
                continue

            # draw 시도
            draw_gaze_vector(frame, (center_x, center_y), gaze_vector, scale=80)

            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                print(f"FPS: {fps:.1f}")
                prev_time = current_time
                frame_count = 0

            print(f"θ: {theta:.2f}, φ: {phi:.2f}, Z: {gaze_z:.2f}, → {status}")
            break

        cv2.imshow("Gaze Visualization", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

main()