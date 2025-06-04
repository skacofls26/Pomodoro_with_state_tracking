'''
config.py
    모델 로드

'''
import mediapipe as mp
from openvino import Core
import os


# ======================= MediaPipe 모델 로딩 (좌표 추출용) =======================

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
FaceMesh = mp_face.FaceMesh(  # 얼굴용 좌표 추출 모델 
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5  
)
Pose = mp_pose.Pose(  # 전신용 좌표 추출 모델 
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3  
)


# ======================= OpenVINO 모델 로딩 (시선 추정용) =======================
def load_OpenVINO_model(core: Core, relative_model_dir: str, device: str = "CPU"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_filename = relative_model_dir + ".xml"
    xml_path = os.path.join(base_dir, xml_filename)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"OpenVINO 모델 XML 파일을 찾을 수 없습니다:\n  예상 경로: {xml_path}")
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, device)
    input_name = compiled.inputs[0].get_any_name()
    return {"model": compiled, "input": input_name}

core = Core()
OpenVINO_models = {
    "face": load_OpenVINO_model(core, "OpenVINO/face-detection-adas-0001/FP32/face-detection-adas-0001"),
    "head_pose": load_OpenVINO_model(core, "OpenVINO/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"),
    "gaze": load_OpenVINO_model(core, "OpenVINO/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"),
}