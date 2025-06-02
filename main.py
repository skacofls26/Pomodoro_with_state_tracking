'''
main.py
    User Interface 구현 로직 (뽀모도로 추가)
    실행 명령어: streamlit run main_base.py
'''

import streamlit as st
import cv2
import time
import pandas as pd
import altair as alt
from datetime import datetime
from config_model import Pose, FaceMesh
from gaze_estimation import is_focused
from drowsy_estimation import is_drowsy
from pomodoro_timer import get_current_phase_duration, transition_to_next_phase

# ========================= 1) Streamlit 페이지 설정 =========================
st.set_page_config(page_title="뽀모도로 타이머", layout="centered")
st.title("🍅 뽀모도로 타이머")

# ========================= 2) 세션 상태 초기화 =========================
if "running" not in st.session_state:
    st.session_state.running = False
if "cap" not in st.session_state:
    st.session_state.cap = None

if "pomodoro_running" not in st.session_state:
    st.session_state.pomodoro_running = False
if "pomodoro_phase" not in st.session_state:
    st.session_state.pomodoro_phase = "idle"
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 1
if "pomodoro_start" not in st.session_state:
    st.session_state.pomodoro_start = None
if "pomodoro_elapsed" not in st.session_state:
    st.session_state.pomodoro_elapsed = 0.0
if "pomodoro_duration_focus" not in st.session_state:
    st.session_state.pomodoro_duration_focus = 25 * 60
if "pomodoro_duration_break" not in st.session_state:
    st.session_state.pomodoro_duration_break = 5 * 60
if "break_long_duration" not in st.session_state:
    st.session_state.break_long_duration = 15 * 60
if "session_count" not in st.session_state:
    st.session_state.session_count = 0

# 시간별 상태 기록용
if "data" not in st.session_state:
    st.session_state.data = {"time": [], "focus": [], "state": []}
if "current_state" not in st.session_state:
    st.session_state.current_state = "비집중 상태"
if "pose_history" not in st.session_state:
    st.session_state.pose_history = []

# 외출/졸음 로그 기록용 리스트
if "state_events" not in st.session_state:
    st.session_state.state_events = []

# 세션별 저장용 리스트
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = []

# EAR(귀 감지) 관련 잠금 플래그
if "state_locked_until_ear_detected" not in st.session_state:
    st.session_state.state_locked_until_ear_detected = False
if "last_known_state_when_no_ear" not in st.session_state:
    st.session_state.last_known_state_when_no_ear = None
if "state_start_time" not in st.session_state:
    st.session_state.state_start_time = None  # 외출/졸음 상태 시작 시각

# 휴식 시작 시각 저장
if "break_start_dt" not in st.session_state:
    st.session_state.break_start_dt = None

# 스냅샷 관련 초기화
if "last_snapshot_time" not in st.session_state:
    st.session_state.last_snapshot_time = 0
if "snapshot_image" not in st.session_state:
    st.session_state.snapshot_image = None
if "snapshot_displayed" not in st.session_state:
    st.session_state.snapshot_displayed = False

# 차트 업데이트 용
if "last_data_len" not in st.session_state:
    st.session_state.last_data_len = 0

# 표시된 사이클/단계 기록 (깜빡임 방지)
if "last_displayed_cycle" not in st.session_state:
    st.session_state.last_displayed_cycle = None
if "last_displayed_phase" not in st.session_state:
    st.session_state.last_displayed_phase = None
    st.session_state.last_snapshot_time = time.monotonic() - 10.0
    st.session_state.snapshot_displayed = False


# ========================= 3) 플레이스홀더 생성 (반드시 사이드바 로직 위에 위치시킬 것) =========================
col_cycle, col_phase = st.columns([1, 1])
with col_cycle:
    cycle_placeholder = st.empty()    # n번째 뽀모도로 표시
with col_phase:
    phase_placeholder = st.empty()    # 단계 표시 ("🔥 집중 단계" 등)

timer_placeholder = st.empty()       # 타이머 표시
if "prev_time_str" not in st.session_state:
    st.session_state.prev_time_str = ""

col_snap, col_chart = st.columns([1, 3])
with col_snap:
    snapshot_placeholder = st.empty()  # 자동 스냅샷 이미지
with col_chart:
    chart_placeholder = st.empty()     # 상태 모니터링 차트

log_placeholder = st.empty()   
pie_placeholder = st.empty()

st.session_state.log_placeholder = log_placeholder
st.session_state.chart_placeholder = chart_placeholder
st.session_state.pie_placeholder = pie_placeholder


# ========================= 4) 사이드바: 뽀모도로 설정 =========================
with st.sidebar:
    st.subheader("시간 설정")
    focus_minutes = st.number_input(
        "집중 시간(분)", min_value=1, max_value=60,
        value=int(st.session_state.pomodoro_duration_focus // 60), step=1
    )
    break_minutes = st.number_input(
        "휴식 시간(분)", min_value=1, max_value=30,
        value=int(st.session_state.pomodoro_duration_break // 60), step=1
    )
    long_break_minutes = st.number_input(
        "긴 휴식 시간(분)", min_value=5, max_value=60,
        value=int(st.session_state.break_long_duration // 60), step=1
    )

    # 입력값을 초 단위로 변환하여 저장
    st.session_state.pomodoro_duration_focus = int(focus_minutes) * 60
    st.session_state.pomodoro_duration_break = int(break_minutes) * 60
    st.session_state.break_long_duration = int(long_break_minutes) * 60

    st.markdown("---")
    st.markdown("#### 타이머 제어")

    # START 버튼: 처음 시작 혹은 재개
    if st.button("▶ START", key="start_pomo"):
        if (
            st.session_state.pomodoro_start is not None
            and not st.session_state.pomodoro_running
        ):
            # 일시 중단된 세션 재개
            st.session_state.pomodoro_start = (
                time.monotonic() - st.session_state.pomodoro_elapsed
            )
            st.session_state.pomodoro_running = True
        else:
            # 새로 세션 시작
            if st.session_state.pomodoro_start is None:
                st.session_state.pomodoro_phase = "focus"
                st.session_state.pomodoro_elapsed = 0.0
                st.session_state.session_count += 1
            st.session_state.pomodoro_start = (
                time.monotonic() - st.session_state.pomodoro_elapsed
            )
            st.session_state.pomodoro_running = True

        # ─── 새 사이클 시작 시 모든 기록·그래프 초기화 ───
        st.session_state.data = {"time": [], "focus": [], "state": []}
        st.session_state.last_data_len = 0
        st.session_state.last_displayed_cycle = None
        st.session_state.last_displayed_phase = None

        # • 로그 초기화
        st.session_state.log_placeholder.empty()
        st.session_state.state_events = []
        st.session_state.last_displayed_event_idx = 0

        # • 휴식 시작 시각 초기화
        st.session_state.break_start_dt = None
        st.session_state.state_start_time = None

        # • 이전 라인 차트 비우기
        st.session_state.chart_placeholder.empty()
        snapshot_placeholder.empty()

        # 카메라 켜기
        if not st.session_state.running:
            st.session_state.cap = cv2.VideoCapture(0)
            if st.session_state.cap.isOpened():
                st.session_state.running = True
                st.session_state.start_time = time.monotonic()
                st.session_state.pose_history = []
                st.session_state.last_snapshot_time = time.monotonic() - 10.0
            else:
                st.error("❌ 카메라 열기 실패")

    # RESTART 버튼: 현재 사이클을 처음부터 다시 수행
    if st.button("↻ RESTART", key="restart_pomo"):
        if st.session_state.pomodoro_start is not None:
            st.session_state.pomodoro_start = time.monotonic()
            st.session_state.pomodoro_elapsed = 0.0
            st.session_state.last_displayed_cycle = None
            st.session_state.last_displayed_phase = None

            # • 로그 초기화
            st.session_state.log_placeholder.empty()
            st.session_state.state_events = []
            st.session_state.last_displayed_event_idx = 0

            # • 휴식 시작 시각 초기화
            st.session_state.break_start_dt = None
            st.session_state.state_start_time = None

            # • 이전 라인 차트 비우기
            # st.session_state.pie_placeholder.empty()
            st.session_state.chart_placeholder.empty()
            snapshot_placeholder.empty()

        # 카메라 재오픈
        st.session_state.cap = cv2.VideoCapture(0)
        if st.session_state.cap.isOpened():
            st.session_state.running = True
            st.session_state.pomodoro_running = True
            st.session_state.start_time = time.monotonic()
            st.session_state.pose_history = []
            st.session_state.last_snapshot_time = time.monotonic() - 10.0
            st.session_state.snapshot_displayed = False
        else:
            st.error("❌ 카메라 열기 실패 (재시작)")

    # STOP 버튼: 세션 정지
    if st.button("■ STOP", key="stop_pomo"):
        if st.session_state.pomodoro_running or st.session_state.running:
            st.session_state.pomodoro_running = False
            st.session_state.running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.write("이어서 실행하려면 ▶ START")
        else:
            st.write("실행 중인 뽀모도로가 없습니다")


# ========================= 5) Main Loop: 카메라 프레임 처리 =========================
if st.session_state.running and st.session_state.cap:
    while st.session_state.running:
        ret, frame = st.session_state.cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        now = time.monotonic()
        H_full, W_full = frame.shape[:2]

        # 해상도 다운스케일 (가로 640 기준)
        WIDTH_SMALL = 640
        HEIGHT_SMALL = int(H_full * WIDTH_SMALL / W_full)
        small = cv2.resize(frame, (WIDTH_SMALL, HEIGHT_SMALL))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Mediapipe 호출
        pose_results = Pose.process(rgb_small)
        face_results = FaceMesh.process(rgb_small)

        # 시선 및 졸음 추정
        try:
            theta, phi, focused = is_focused(small)
        except RuntimeError:
            focused = False
        ear, drowsy = is_drowsy(
            face_results.multi_face_landmarks, HEIGHT_SMALL, WIDTH_SMALL
        )

        # ========================= a. 데이터 저장 =========================
        timestamp = datetime.now()
        st.session_state.data["time"].append(timestamp)
        st.session_state.data["focus"].append("Focused" if focused else "Not Focused")
        st.session_state.data["state"].append(st.session_state.current_state)

        # Pose 히스토리 누적 (외출/졸음이 아닐 때만)
        if (
            st.session_state.current_state not in ["졸음 상태", "외출 상태"]
            and pose_results
            and pose_results.pose_landmarks
        ):
            left = pose_results.pose_landmarks.landmark[7]
            right = pose_results.pose_landmarks.landmark[8]
            ys = []
            if left.visibility > 0.5:
                ys.append(left.y * HEIGHT_SMALL)
            if right.visibility > 0.5:
                ys.append(right.y * HEIGHT_SMALL)
            if ys:
                avg_y = sum(ys) / len(ys)
                st.session_state.pose_history.append((now, avg_y))
                st.session_state.pose_history = [
                    (t, y) for (t, y) in st.session_state.pose_history if now - t <= 3.0
                ]

        # ========================= b. 상태 판별 =========================
        new_state = None

        # EAR 재감지 시: 외출/졸음 상태 유지 시간 기록
        if ear is not None and st.session_state.state_locked_until_ear_detected:
            prev = st.session_state.last_known_state_when_no_ear
            # 5초 이상 머물렀다면 기록
            elapsed_state = now - st.session_state.state_start_time
            if prev in ["외출 상태", "졸음 상태"] and elapsed_state >= 5.0:
                # 시/분/초 단위로 분해
                hours = int(elapsed_state // 3600)
                minutes = int((elapsed_state % 3600) // 60)
                seconds = int(elapsed_state % 60)
                parts = []
                if hours > 0:
                    parts.append(f"{hours}시간")
                if minutes > 0:
                    parts.append(f"{minutes}분")
                parts.append(f"{seconds}초")
                elapsed_str = " ".join(parts)

                # 현재 시각을 [HH시 MM분 SS초] 형식으로
                now_dt = datetime.now()
                ts_h = now_dt.hour
                ts_m = now_dt.minute
                ts_s = now_dt.second
                ts_formatted = f"[{ts_h:02d}시 {ts_m:02d}분 {ts_s:02d}초]"

                # 로그 메시지 추가
                log_msg = f"{ts_formatted} {elapsed_str} 간 {prev}였습니다."
                st.session_state.state_events.append(
                    {"timestamp": ts_formatted, "message": log_msg}
                )

            # 잠금 해제
            st.session_state.state_locked_until_ear_detected = False
            st.session_state.last_known_state_when_no_ear = None
            st.session_state.state_start_time = None

        # EAR 로 신호가 들어오면 일반 판별
        if st.session_state.state_locked_until_ear_detected:
            # 잠금 상태면 이전 상태 유지
            new_state = st.session_state.last_known_state_when_no_ear
        else:
            if drowsy:
                new_state = "졸음 상태"
            elif focused:
                new_state = "집중 상태"
                st.session_state.focus_timer = None
            else:
                if ear is None:
                    ear_y_values = [y for (_, y) in st.session_state.pose_history]
                    if len(ear_y_values) < 10:
                        new_state = st.session_state.current_state
                    else:
                        y_diff = max(ear_y_values) - min(ear_y_values)
                        if y_diff >= 15:
                            max_y_time = next(
                                t
                                for (t, y) in st.session_state.pose_history
                                if y == max(ear_y_values)
                            )
                            min_y_time = next(
                                t
                                for (t, y) in st.session_state.pose_history
                                if y == min(ear_y_values)
                            )
                            if max_y_time < min_y_time:
                                result = "외출 상태"
                            else:
                                result = "졸음 상태"
                        else:
                            result = "비집중 상태"

                        # 외출/졸음 상태로 진입할 때
                        if result in ["외출 상태", "졸음 상태"]:
                            st.session_state.last_known_state_when_no_ear = result
                            st.session_state.state_locked_until_ear_detected = True
                            st.session_state.state_start_time = now  # 상태 시작 시각 저장
                            new_state = result
                        else:
                            new_state = "비집중 상태"
                else:
                    new_state = "비집중 상태"

        st.session_state.current_state = new_state

        # ========================= c. UI 업데이트 =========================

        # 1) n번째 뽀모도로
        cyc = st.session_state.cycle_count
        if st.session_state.last_displayed_cycle != cyc:
            cycle_placeholder.markdown(
                f"<div style='background-color:rgba(255, 243, 205, 0.6); padding:4px; border-radius:4px; margin-bottom:0;'>"
                f"<span style='font-size:16px; font-weight:600;'>🍅 사이클:&nbsp;&nbsp;{cyc}회</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.session_state.last_displayed_cycle = cyc

        # 2) 단계 표시 및 뽀모도로 요약
        ph = st.session_state.pomodoro_phase
        if st.session_state.last_displayed_phase != ph:
            if ph in ["break_short", "break_long"]:
                st.session_state.pie_placeholder.empty()

                df_all = pd.DataFrame({"State": st.session_state.data["state"]})
                counts = df_all["State"].value_counts().reset_index()
                counts.columns = ["State", "Count"]
                total = counts["Count"].sum()
                counts["Ratio"] = counts["Count"] / total

                hover = alt.selection_point(
                    fields=["State"],     
                    on="mouseover",
                    clear="mouseout"
                )

                pie = (
                    alt.Chart(counts)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="State", type="nominal", 
                            scale=alt.Scale(
                                domain=["집중 상태", "비집중 상태", "외출 상태", "졸음 상태"],
                                range=["#4CAF50", "#F44336", "#FFC107", "#2196F3"]
                            ),
                            legend=alt.Legend(title="상태")
                        ),
                        tooltip=[
                            alt.Tooltip("State:N", title="상태"),
                            alt.Tooltip("Ratio:Q", format=".1%", title="비율")
                        ],
                        opacity=alt.condition(hover, alt.value(1), alt.value(0.6)),
                    )
                    .add_params(hover)  # hover selection_point을 차트에 추가
                    .properties(
                        title=f"이전 뽀모도로 요약",
                        width=200,
                        height=200
                    )
                )
                st.session_state.pie_placeholder.altair_chart(pie, use_container_width=True)
            
            if ph == "focus":
                st.session_state.state_events = []
                st.session_state.last_displayed_event_idx = 0
                st.session_state.log_placeholder.empty()

                phase_placeholder.markdown(
                    "<div style='background-color:rgba(212, 237, 218, 0.6);; padding:4px; border-radius:4px; "
                    "margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>🔥 집중 단계</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif ph == "break_short":
                phase_placeholder.markdown(
                    "<div style='background-color:rgba(204, 229, 255, 0.6); padding:4px; border-radius:4px; "
                    "margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>☕ 짧은 휴식 단계</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif ph == "break_long":
                phase_placeholder.markdown(
                    "<div style='background-color:rgba(226, 221, 236, 0.6); padding:4px; border-radius:4px; "
                    "margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>💤 긴 휴식 단계</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                phase_placeholder.empty()
            st.session_state.last_displayed_phase = ph

        # 3) 타이머 표시
        if st.session_state.pomodoro_running:
            total = get_current_phase_duration()
            elapsed = now - st.session_state.pomodoro_start

            if elapsed < total:
                rem = total - elapsed
                m, s = divmod(int(rem), 60)
                curr_time_str = f"{m:02d}:{s:02d}"
                if curr_time_str != st.session_state.prev_time_str:
                    timer_placeholder.markdown(
                        f"<div style='text-align:left; margin:0;'>"
                        f"<span style='font-size:120px; font-weight:600;'>{curr_time_str}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.session_state.prev_time_str = curr_time_str
                st.session_state.pomodoro_elapsed = elapsed
            else:
                transition_to_next_phase()
                st.session_state.prev_time_str = ""
        else:
            timer_placeholder.empty()
            st.session_state.prev_time_str = ""

        # ========================= d. 자동 스냅샷 =========================
        elapsed_snap = now - st.session_state.last_snapshot_time
        if elapsed_snap > 10:
            st.session_state.snapshot_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_snapshot_time = now
            st.session_state.snapshot_displayed = True

            ts = datetime.fromtimestamp(st.session_state.last_snapshot_time).strftime("%H:%M:%S")
            snapshot_placeholder.subheader(f"📸 {ts}")
            snapshot_placeholder.image(st.session_state.snapshot_image, width=200)
        else:
            if elapsed_snap > 12 and st.session_state.snapshot_displayed:
                snapshot_placeholder.empty()
                st.session_state.snapshot_displayed = False

        # ========================= e. 외출/졸음 로그 출력 =========================
        if st.session_state.state_events:
            if "last_displayed_event_idx" not in st.session_state:
                st.session_state.last_displayed_event_idx = 0
            for evt in st.session_state.state_events[st.session_state.last_displayed_event_idx:]:
                st.write(evt["message"])
                st.session_state.last_displayed_event_idx += 1

        # ========================= f. 상태 모니터링 차트 =========================
        current_len = len(st.session_state.data["time"])
        if current_len > st.session_state.last_data_len:
            df = pd.DataFrame({
                "Time": st.session_state.data["time"],
                "State": st.session_state.data["state"]
            })
            if not df.empty:
                tip = df["Time"].max()
                recent = df[df["Time"] >= tip - pd.Timedelta(seconds=30)].copy()
                recent["State_Visual"] = recent["State"].replace({"수면 상태": "졸음 상태"})
                chart = (
                    alt.Chart(recent)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X("Time:T", title="Time"),
                        y=alt.Y("State", title="State"),
                        color=alt.Color(
                            "State_Visual",
                            legend=None,
                            scale=alt.Scale(
                                domain=[
                                    "집중 상태",
                                    "졸음 상태",
                                    "외출 상태",
                                    "비집중 상태",
                                ],
                                range=["green", "blue", "orange", "red"],
                            ),
                        ),
                        tooltip=["Time", "State"],
                    )
                    .properties(height=150)
                )
                st.session_state.chart_placeholder.altair_chart(chart, use_container_width=True)
            else:
                st.session_state.chart_placeholder.empty()

            st.session_state.last_data_len = current_len

        # 속도 유지 (약 5fps)
        time.sleep(0.2)
        if not st.session_state.running:
            break

elif not st.session_state.running:
    pass