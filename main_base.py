'''
main.py
    User Interface 구현 로직
    실행 명령어: streamlit run main_base.py
'''
import streamlit as st
import cv2
import time
import pandas as pd
import altair as alt
from datetime import datetime
from config import Pose, FaceMesh
from gaze_estimation import is_focused
from drowsy_estimation import is_drowsy

st.set_page_config(page_title="학습 보조 애플리케이션", layout="centered")
st.title("\U0001f9e0 학습 보조 애플리케이션")

# ========================= 상태 초기화 =========================
if "cap" not in st.session_state:
    st.session_state.cap = None
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = {
        "time": [], "focus": [], "state": []
    }
if "current_state" not in st.session_state:
    st.session_state.current_state = "화면 비집중 상태"
if "focus_timer" not in st.session_state:
    st.session_state.focus_timer = None
if "snapshot_image" not in st.session_state:
    st.session_state.snapshot_image = None
if "last_snapshot_time" not in st.session_state:
    st.session_state.last_snapshot_time = 0
if "drowsy" not in st.session_state:
    st.session_state.drowsy = False
if "eye_closed_start" not in st.session_state:
    st.session_state.eye_closed_start = None
if "last_known_state_when_no_ear" not in st.session_state:
    st.session_state.last_known_state_when_no_ear = None
if "state_locked_until_ear_detected" not in st.session_state:
    st.session_state.state_locked_until_ear_detected = False
if "pose_history" not in st.session_state:
    st.session_state.pose_history = []  # [(time, left_ear_y, right_ear_y)]

# ✅ 스트리밍 버튼
with st.container():
    col_btn = st.columns([1])[0]
    toggle_label = "⏹ 스트리밍 중지" if st.session_state.running else "▶ 스트리밍 시작"
    if col_btn.button(toggle_label, key="stream_button"):
        if st.session_state.running:
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.running = False
            st.rerun()
        else:
            st.session_state.cap = cv2.VideoCapture(0)
            if st.session_state.cap.isOpened():
                st.session_state.running = True
                st.session_state.start_time = time.monotonic()
                st.session_state.focus_timer = None
                st.session_state.last_remain_sec = None
                st.session_state.drowsy = False
                st.session_state.eye_closed_start = None
                st.session_state.last_known_state_when_no_ear = None
                st.session_state.state_locked_until_ear_detected = False
                st.session_state.pose_history = []
            else:
                st.error("❌ 카메라 열기에 실패했습니다.")

# ========================= 실시간 스트리밍 =========================
if st.session_state.running and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if not ret:
        st.warning("❌ 프레임을 읽을 수 없습니다.")
    else:
        now = time.monotonic() 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = Pose.process(rgb)
        face_results = FaceMesh.process(rgb)
        h, w = frame.shape[:2]

        theta, phi, focused = is_focused(frame)        
        ear, drowsy = is_drowsy(face_results.multi_face_landmarks, h, w)

        timestamp = datetime.now()

        # ================= Pose 히스토리 누적 =================
        if st.session_state.current_state not in ["수면 상태", "외출 상태"]: 
            if pose_results and pose_results.pose_landmarks:
                left_ear = pose_results.pose_landmarks.landmark[7]
                right_ear = pose_results.pose_landmarks.landmark[8]
                y_vals = []
                if left_ear.visibility > 0.5:
                    y_vals.append(left_ear.y * h)
                if right_ear.visibility > 0.5:
                    y_vals.append(right_ear.y * h)
                if y_vals:
                    avg_y = sum(y_vals) / len(y_vals)
                    st.session_state.pose_history.append((now, avg_y))
                    st.session_state.pose_history = [
                        (t, y) for (t, y) in st.session_state.pose_history if now - t <= 3.0
                    ]

        # ================= 상태 판별 =================
        new_state = None

        # ─── 0) EAR 감지 시 초기화 ───
        if ear is not None:
            st.session_state.state_locked_until_ear_detected = False
            st.session_state.last_known_state_when_no_ear = None

        if st.session_state.state_locked_until_ear_detected:  # 수면/외출 판단했으면 유지
            new_state = st.session_state.last_known_state_when_no_ear
        else:
            # ─── 1) 졸음 판단 ───
            if drowsy:                                
                new_state = "졸음 상태"  # drowsy는 집중 상태에서도 나타나므로 먼저 판단
            
            # ─── 2) 화면 집중 ───
            elif focused:
                new_state = "화면 집중 상태"
                st.session_state.focus_timer = None 

            # ─── 3) 화면 비집중 ───
            else:

                # ─── 3-1) EAR 미감지 ───
                if ear is None:
                    ear_y_values = [y for (_, y) in st.session_state.pose_history]
                    if len(ear_y_values) < 10:   
		                # 최근 3초 기록에서 데이터 부족 
                        new_state = st.session_state.current_state
                    else:
                        # 1회 판단
                        y_diff = max(ear_y_values) - min(ear_y_values)
                        if y_diff >= 15:
                            # max(y_values)가 min(y_values)보다 나중이면 수면, 먼저면 외출
                            max_y_time = next(t for (t, y) in st.session_state.pose_history if y == max(ear_y_values))
                            min_y_time = next(t for (t, y) in st.session_state.pose_history if y == min(ear_y_values))
                            if max_y_time < min_y_time:
                                result = "외출 상태"
                            else:
                                result = "수면 상태"
                        else:
                            # 별 차이 안나면 비집중
                            result = "화면 비집중 상태"
                            
                        # EAR 재감지 전까지 잠금 
                        st.session_state.last_known_state_when_no_ear = result
                        st.session_state.state_locked_until_ear_detected = True
                        new_state = result
                # ─── 3-2) EAR 감지 ───
                else:
                    new_state = "화면 비집중 상태"

        # ─── 4) 현재 상태 저장 ───
        st.session_state.current_state = new_state

        # 현재 상태 표시
        with st.container():
            state_text = st.session_state.current_state
            if "집중" in state_text and "비집중" not in state_text:
                st.success(f"🧠  {state_text}")
            else:
                st.error(f"🧠  {state_text}")

        # 데이터 저장
        st.session_state.data["time"].append(timestamp)
        st.session_state.data["focus"].append("Focused" if focused else "Not Focused")
        st.session_state.data["stte"].append(st.session_state.current_state)

        # 자동 스냅샷
        if now - st.session_state.last_snapshot_time > 10:
            st.session_state.snapshot_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_snapshot_time = now

        # 시각화: 스냅샷 + 집중 상태
        show_snapshot = (
            st.session_state.snapshot_image is not None and
            time.monotonic() - st.session_state.last_snapshot_time < 12
        )

        if show_snapshot or st.session_state.data["time"]:
            with st.container():
                col_snap, col_chart = st.columns([1, 2])

                with col_snap:
                    if show_snapshot:
                        ts_txt = datetime.fromtimestamp(
                            st.session_state.last_snapshot_time
                        ).strftime('%H:%M:%S')
                        st.subheader(f"📸 {ts_txt}")
                        st.image(st.session_state.snapshot_image, width=200)

                with col_chart:
                    df = pd.DataFrame({
                        "Time": st.session_state.data["time"],
                        "State": st.session_state.data["state"]
                    })
                    latest_time = df["Time"].max()
                    df_recent = df[df["Time"] >= latest_time - pd.Timedelta(seconds=30)]
                    df_recent["State_Visual"] = df_recent["State"].replace({
                        "수면 상태": "졸음 상태"
                    })
                    
                    st.subheader("📉 상태 모니터링")
                    focus_chart = alt.Chart(df_recent).mark_circle(size=60).encode(
                        x=alt.X("Time:T", title="Time"),
                        y=alt.Y("State", title="State"),
                        color=alt.Color("State_Visual", legend=None, scale=alt.Scale(
                            domain=[
                                "화면 집중 상태",
                                "졸음 상태",
                                "외출 상태",
                                "화면 비집중 상태"
                            ],
                            range=["green", "blue", "orange", "red"]
                        )),
                        tooltip=["Time", "State"]
                    ).properties(height=150)
                    st.altair_chart(focus_chart, use_container_width=True)

# =================== 자동 반복 ===================
if st.session_state.running:
    time.sleep(0.1)
    st.rerun()
