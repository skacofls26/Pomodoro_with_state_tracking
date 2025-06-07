'''
drowsy_estimation.py
    EAR 기반 졸음 판단 로직
'''
import streamlit as st
import numpy as np
import time
from scipy.spatial import distance as dist

# ======================= 상수 정의 =======================

# 눈 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 졸음 판단 임계치
EAR_THRESH = 0.21  # 눈 감은 정도에 대한 임계치
DROWSY_THRESH = 4.0  # 졸음 지속 시간(초)에 대한 임계치 

# ======================= 보조 함수 =======================
def compute_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# ======================= EAR 기반 졸음 판단 함수 =======================
def is_drowsy(FaceMesh_landmarks, h, w):
    if not FaceMesh_landmarks:
        return None, False

    mesh = FaceMesh_landmarks[0].landmark
    left_eye = np.array([(mesh[i].x * w, mesh[i].y * h) for i in LEFT_EYE_IDX])
    right_eye = np.array([(mesh[i].x * w, mesh[i].y * h) for i in RIGHT_EYE_IDX])

    ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
    now = time.monotonic()

    # 상태 초기화
    if "drowsy" not in st.session_state:
        st.session_state.drowsy = False
    if "eye_closed_start" not in st.session_state:
        st.session_state.eye_closed_start = None

    if ear < EAR_THRESH:
        if st.session_state.eye_closed_start is None:
            st.session_state.eye_closed_start = now
        elif not st.session_state.drowsy and (now - st.session_state.eye_closed_start) >= DROWSY_THRESH:
            st.session_state.drowsy = True
    else:
        st.session_state.eye_closed_start = None
        st.session_state.drowsy = False

    return ear, st.session_state.drowsy
