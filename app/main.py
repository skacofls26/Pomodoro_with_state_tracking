'''
main.py
    User Interface 구현 로직 (뽀모도로)
    실행 명령어: streamlit run main.py
'''
import logging
import streamlit as st
import cv2
import numpy as np 
import time
import pandas as pd
import altair as alt
from datetime import datetime
from pathlib import Path
from config_model import FaceMesh
from gaze_estimation import is_focused
from drowsy_estimation import is_drowsy
from pomodoro_timer import (
    get_current_phase_duration,
    transition_to_next_phase,
    draw_break_long_pies,
)
import queue

def main():
    # ─── 0) 로그 레벨 조정 ─────────────────────────────────────
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

    # ─────────────────────────── 1) 페이지·헤더 ────────────────────────────
    st.set_page_config(page_title="뽀모도로 타이머", layout="centered")
    st.title("🍅 뽀모도로 타이머")

    if "congrats_msg" not in st.session_state:
        st.session_state.congrats_msg = ""

    if st.session_state.congrats_msg:
        st.markdown(
            f"<p style='font-size:18px; font-weight:600; color:#000000; margin:0'>"
            f"{st.session_state.congrats_msg}"
            f"</p>",
            unsafe_allow_html=True,
        )

    # ─────────────────────────── 2) 세션 상태 초기화 ──────────────────────
    def _init_state(key, default):
        if key not in st.session_state:
            st.session_state[key] = default

    _init_state("running", False)           # 카메라+타이머 작동 여부
    _init_state("pomodoro_running", False)  # 타이머만 작동 여부
    _init_state("pomodoro_phase", "focus")
    _init_state("cycle_count", 1)
    _init_state("pomodoro_start", None)
    _init_state("pomodoro_elapsed", 0.0)
    _init_state("pomodoro_duration_focus", 25 * 60)
    _init_state("pomodoro_duration_break", 5 * 60)
    _init_state("break_long_duration", 15 * 60)

    # ← 짧은 휴식 파이 그리기 여부를 저장할 플래그
    _init_state("break_short_drawn", False)

    # ← 직전 프레임에서 어떤 phase였는지 저장
    _init_state("last_displayed_phase", None)

    _init_state("session_count", 0)
    _init_state("break_long_drawn", False)  # 긴 휴식 pie 3개 이미 그렸는가
    _init_state("prev_time_str", "")

    # 기록용 컨테이너들
    _init_state("data", {"time": [], "focus": [], "state": []})
    _init_state("current_state", "비집중 상태")
    _init_state("left_ear_history", [])
    _init_state("right_ear_history", [])
    _init_state("state_events", [])
    _init_state("all_sessions", [])

    # 잠금 관련
    _init_state("state_locked_until_ear_detected", False)
    _init_state("last_known_state_when_no_ear", None)
    _init_state("state_start_time", None)
    _init_state("detect_prev_state", None)
    _init_state("detect_start_time", None)

    # 기타
    _init_state("break_start_dt", None)
    _init_state("last_snapshot_time", 0.0)
    _init_state("snapshot_displayed", False)
    _init_state("last_data_len", 0)
    _init_state("last_displayed_cycle", None)
    _init_state("last_displayed_phase", None)

    _init_state("congrats_msg", "")
    _init_state("snap_img_elem", None)


    # ─────────────────────────── 3) 플레이스홀더 ───────────────────────────
    col_cycle, col_phase = st.columns([1, 1])
    cycle_placeholder  = col_cycle.empty()
    phase_placeholder  = col_phase.empty()
    timer_placeholder  = st.empty()

    # 스냅샷 전용 컬럼(왼쪽)
    col_snap, col_chart = st.columns([1, 3])
    snap_box = col_snap.container()
    chart_placeholder = col_chart.empty()

    if st.session_state.snap_img_elem is None:
        transparent_px = np.zeros((1, 1, 4), dtype=np.uint8)
        st.session_state.snap_img_elem = snap_box.image(
            transparent_px, channels="RGBA", use_container_width="auto"
        )

    log_placeholder    = st.empty()
    pie_placeholder    = st.empty()
    download_pl        = st.empty()

    st.session_state.log_placeholder   = log_placeholder
    st.session_state.chart_placeholder = chart_placeholder
    st.session_state.pie_placeholder   = pie_placeholder


    # ─────────────────────────── 4) 사이드바 ───────────────────────────────
    with st.sidebar:
        st.subheader("시간 설정")
        focus_minutes = st.number_input(
            "집중 시간(분)", 1, 60,
            int(st.session_state.pomodoro_duration_focus // 60)
        )
        break_minutes = st.number_input(
            "휴식 시간(분)", 1, 30,
            int(st.session_state.pomodoro_duration_break // 60)
        )
        long_break_minutes = st.number_input(
            "긴 휴식 시간(분)", 5, 60,
            int(st.session_state.break_long_duration // 60)
        )

        st.session_state.pomodoro_duration_focus = focus_minutes * 60
        st.session_state.pomodoro_duration_break = break_minutes * 60
        st.session_state.break_long_duration     = long_break_minutes * 60

        st.markdown("---")
        st.markdown("#### 타이머 제어")

        # ▶ START
        if st.button("▶ START", key="start_pomo"):
            st.empty()
            st.session_state.congrats_msg = ""
            st.session_state.pop("download_btn_drawn", None)

            if st.session_state.pomodoro_start is not None and not st.session_state.pomodoro_running:
                # 일시정지 후 재개
                st.session_state.pomodoro_start = (
                    time.monotonic() - st.session_state.pomodoro_elapsed
                )
            else:
                # 새 세션 시작
                st.session_state.pomodoro_phase   = "focus"
                st.session_state.pomodoro_elapsed = 0.0
                st.session_state.session_count   += 1
                st.session_state.pomodoro_start   = time.monotonic()

                # 기록 초기화
                st.session_state.data = {"time": [], "focus": [], "state": []}
                st.session_state.state_events.clear()
                st.session_state.break_long_drawn = False

                # 파이·로그·차트 초기화
                pie_placeholder.empty()
                log_placeholder.empty()
                chart_placeholder.empty()

            st.session_state.pomodoro_running = True
            st.session_state.running          = True
            st.session_state.start_time       = time.monotonic()

            # 새 사이클이 시작될 때 짧은 휴식 캐시 초기화
            st.session_state.break_short_drawn = False
            st.session_state.pie_placeholder.empty()


        # ⏸ PAUSE
        if st.button("⏸ PAUSE", key="pause_pomo"):
            if st.session_state.pomodoro_running:
                st.session_state.pomodoro_elapsed = (
                    time.monotonic() - st.session_state.pomodoro_start
                )
                st.session_state.pomodoro_running = False
                st.session_state.running          = False

                st.write("이어서 실행하려면 ▶ START")
            else:
                st.write("진행 중인 뽀모도로 없음")


        # ↻ RESET
        if st.button("↻ RESET", key="restart_pomo"):
            st.empty()
            st.session_state.congrats_msg = ""
            st.session_state.pop("download_btn_drawn", None)
            download_pl.empty()

            st.session_state.pomodoro_start = time.monotonic()
            st.session_state.pomodoro_elapsed = 0.0
            st.session_state.pomodoro_running = True
            st.session_state.running          = True
            st.session_state.break_long_drawn = False

            # 파이·로그·차트 초기화
            pie_placeholder.empty()
            log_placeholder.empty()
            chart_placeholder.empty()
            st.session_state.data = {"time": [], "focus": [], "state": []}
            st.session_state.state_events.clear()

            # 짧은 휴식 캐시 초기화
            st.session_state.break_short_drawn = False
            st.session_state.pie_placeholder.empty()


        # ■ STOP
        if st.button("■ STOP", key="stop_pomo"):
            st.empty()
            st.session_state.break_long_drawn = False
            download_pl.empty()
            st.session_state.pop("download_btn_drawn", None)

            if st.session_state.pomodoro_running or st.session_state.running:
                st.session_state.pomodoro_running = False
                st.session_state.running = False
                st.session_state.pomodoro_start = None
                st.session_state.pomodoro_elapsed = 0.0

                # 🔧 추가된 초기화
                st.session_state.pomodoro_phase = "focus"
                st.session_state.cycle_count = 1
                st.session_state.session_count = 0
                st.session_state.last_displayed_phase = None
                st.session_state.prev_time_str = ""
                
                # 완료한 사이클 계산
                phase = st.session_state.pomodoro_phase
                if phase == "focus":
                    cycles_done = st.session_state.cycle_count - 1
                else:
                    cycles_done = st.session_state.cycle_count
                    
                if cycles_done > 0:
                    st.session_state.congrats_msg = (
                        f"오늘은 {cycles_done}번째 사이클까지 클리어하셨습니다 🎉"
                    )
            else:
                st.write("실행 중인 뽀모도로가 없습니다")


    # ========================= 4.5) WebRTC 스트림 (UI 숨김) =========================
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    webrtc_ctx = webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDONLY,
        desired_playing_state=st.session_state.running,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"style": {"display": "none", "width": "0px", "height": "0px"}},
    )

    st.markdown(
        """
        <style>
        /* 재생(START)·정지(STOP) 버튼 숨김 */
        button[title="Start"], button[title="Stop"] {
            display:none !important;
        }
        /* 연결 상태(●, “connecting…” 등) 숨김 */
        div.st-webrtc-status {
            display:none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # ========================= 5) Main Loop: WebRTC 프레임 처리 =========================
    if st.session_state.running and webrtc_ctx and webrtc_ctx.state.playing:

        while st.session_state.running and webrtc_ctx.state.playing:
            if webrtc_ctx.video_receiver is None:
                time.sleep(0.05)
                continue

            # ------------- 프레임 수신 -------------
            try:
                av_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                continue

            if av_frame is None:
                continue

            # av.VideoFrame → numpy(BGR)
            frame_bgr = av_frame.to_ndarray(format="bgr24")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            now = time.monotonic()
            H_full, W_full = frame_rgb.shape[:2]

            # 해상도 다운스케일 (가로 640 기준)
            WIDTH_SMALL = 320
            HEIGHT_SMALL = int(H_full * WIDTH_SMALL / W_full)
            small = cv2.resize(frame_rgb, (WIDTH_SMALL, HEIGHT_SMALL))
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Mediapipe 호출
            face_results = FaceMesh.process(rgb_small)

            # 시선 및 졸음 추정
            try:
                theta, phi, focused = is_focused(small)
            except RuntimeError:
                focused = False
            ear, drowsy = is_drowsy(
                face_results.multi_face_landmarks, HEIGHT_SMALL, WIDTH_SMALL
            )

            # 히스토리 누적 (외출/졸음이 아닐 때만)
            if (
                st.session_state.current_state not in ["졸음 상태", "외출 상태"]
                and face_results
                and face_results.multi_face_landmarks
            ):
                landmarks = face_results.multi_face_landmarks[0].landmark
                try:
                    lx = landmarks[234].x * WIDTH_SMALL
                    ly = landmarks[234].y * HEIGHT_SMALL
                    st.session_state.left_ear_history.append((now, lx, ly))
                except:
                    pass
                try:
                    rx = landmarks[454].x * WIDTH_SMALL
                    ry = landmarks[454].y * HEIGHT_SMALL
                    st.session_state.right_ear_history.append((now, rx, ry))
                except:
                    pass
                st.session_state.left_ear_history = [
                    (t, x, y) for (t, x, y) in st.session_state.left_ear_history if now - t <= 3.0
                ]
                st.session_state.right_ear_history = [
                    (t, x, y) for (t, x, y) in st.session_state.right_ear_history if now - t <= 3.0
                ]
                # print(f"(L: {lx:.3f}, {ly:.3f}) (R: {rx:.3f}, {ry:.3f})\n")

            # ========================= a. 데이터 저장 =========================
            timestamp = datetime.now()
            st.session_state.data["time"].append(timestamp)
            st.session_state.data["focus"].append("Focused" if focused else "Not Focused")
            st.session_state.data["state"].append(st.session_state.current_state)

            MAX_HISTORY = 60 * 5
            for k in ("time", "state", "focus"):
                if len(st.session_state.data[k]) > MAX_HISTORY:
                    st.session_state.data[k] = st.session_state.data[k][-MAX_HISTORY:]


            # ========================= b. 상태 판별 =========================
            # ─── 0) 새로운 상태 변수 초기화 ───
            new_state = None

            # ─── 1) EAR 재감지 시 잠금 해제 ───
            if ear is not None and st.session_state.state_locked_until_ear_detected:
                st.session_state.state_locked_until_ear_detected = False
                st.session_state.last_known_state_when_no_ear = None
                st.session_state.state_start_time = None
            
            # ─── 2) (졸음/외출로 인한 잠금 상태) 이전 상태 유지 ───
            if st.session_state.state_locked_until_ear_detected:
                new_state = st.session_state.last_known_state_when_no_ear
            else:
                # ─── 3) 졸음 상태 판단 ───
                if drowsy:
                    new_state = "졸음 상태"
                # ─── 4) 집중 상태 판단 ───
                elif focused:
                    new_state = "집중 상태"
                    st.session_state.focus_timer = None
                else:
                    # ─── 5) EAR 미감지 시 상태 판단 ───
                    if ear is None:
                        # 가장 최근 EAR y값 가져오기
                        left_ear_vals  = [(x, y) for (_, x, y) in st.session_state.left_ear_history][-10:]
                        right_ear_vals = [(x, y) for (_, x, y) in st.session_state.right_ear_history][-10:]

                        if len(left_ear_vals) >= 4 and len(right_ear_vals) >= 4:
                            before_left_y  = sum([y for (_, y) in left_ear_vals[:3]]) / 3
                            after_left_y   = left_ear_vals[-1][1]
                            left_y_diff = after_left_y - before_left_y

                            before_right_y = sum([y for (_, y) in right_ear_vals[:3]]) / 3
                            after_right_y  = right_ear_vals[-1][1]
                            right_y_diff = after_right_y - before_right_y

                            before_left_x  = sum([x for (x, _) in left_ear_vals[:3]]) / 3
                            after_left_x   = left_ear_vals[-1][0]
                            left_x_diff = after_left_x - before_left_x

                            before_right_x = sum([x for (x, _) in right_ear_vals[:3]]) / 3
                            after_right_x  = right_ear_vals[-1][0]
                            right_x_diff = after_right_x - before_right_x

                            max_x_diff = max(abs(left_x_diff), abs(right_x_diff))

                            if abs(left_y_diff) < 3 and abs(right_y_diff) < 3 and max_x_diff < 5:
                                result = "비집중 상태"
                            elif ((abs(left_y_diff) > abs(right_y_diff) and left_y_diff < 0) or 
                                    (abs(left_y_diff) < abs(right_y_diff) and right_y_diff < 0)):
                                result = "외출 상태"  # 귀 y값 올라감 → 외출
                            else:
                                result = "졸음 상태"   # 귀 y값 내려감 → 졸음

                            # 외출/졸음 상태 진입 시 잠금 및 시간 기록
                            if result in ["외출 상태", "졸음 상태", "비집중 상태"]:
                                st.session_state.last_known_state_when_no_ear = result
                                st.session_state.state_locked_until_ear_detected = True
                                st.session_state.state_start_time = now  
                                new_state = result
                            else:
                                new_state = "비집중 상태"
                            # if abs(left_y_diff) > abs(right_y_diff):
                            #     formatted = [f"{y:.1f}" for (_, y) in left_ear_vals]
                            #     print(f"y history: {formatted}")
                            # else:
                            #     formatted = [f"{y:.1f}" for (_, y) in right_ear_vals]
                            #     print(f"y history: {formatted}")
                        else:
                            new_state = "비집중 상태"  # 데이터 부족
                     # ─── 6) EAR 감지 시 비집중 상태 ───
                    else:
                        new_state = "비집중 상태"  

            st.session_state.current_state = new_state

            # ────── 상태 전환 감지 → 외출/졸음 이벤트 기록 ──────
            prev = st.session_state.detect_prev_state
            curr = new_state
            if curr in ["외출 상태", "졸음 상태"] and prev != curr:
                st.session_state.detect_start_time = now
            
            elif prev in ["외출 상태", "졸음 상태"] and curr != prev:
                elapsed = now - st.session_state.detect_start_time
                if elapsed >= 5.0:
                    hrs = int(elapsed // 3600)
                    mins = int((elapsed % 3600) // 60)
                    secs = int(elapsed % 60)
                    parts = []
                    if hrs:   parts.append(f"{hrs}시간")
                    if mins:  parts.append(f"{mins}분")
                    parts.append(f"{secs}초")
                    elapsed_str = " ".join(parts)

                    ts = datetime.now()
                    ts_fmt = f"[{ts.hour:02d}시 {ts.minute:02d}분 {ts.second:02d}초]"
                    st.session_state.state_events.append({
                        "timestamp": ts_fmt,
                        "message": f"{ts_fmt} {elapsed_str} 간 {prev}였습니다."
                    })
             # 이전 상태 업데이트
            st.session_state.detect_prev_state = curr


            # ========================= c. UI 업데이트 =========================

            # ────────────────────────────── 1) 사이클 표시 ──────────────────────────────
            cyc = st.session_state.cycle_count
            cycle_placeholder.markdown(
                f"<div style='background-color:rgba(255, 243, 205, 0.6); "
                f"padding:4px; border-radius:4px; margin-bottom:0;'>"
                f"<span style='font-size:16px; font-weight:600;'>"
                f"♻️ &nbsp;사이클:&nbsp;&nbsp;{cyc}번째</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ────────────────────────────── 2) 단계 텍스트 표시 ──────────────────────────────
            ph = st.session_state.pomodoro_phase
            if ph == "focus":
                phase_html = (
                    "<div style='background-color:rgba(212, 237, 218, 0.6); padding:4px; "
                    "border-radius:4px; margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>🔥 집중 시간</span>"
                    "</div>"
                )
            elif ph == "break_short":
                phase_html = (
                    "<div style='background-color:rgba(255, 204, 204, 0.6); padding:4px; "
                    "border-radius:4px; margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>☕ 짧은 휴식 시간</span>"
                    "</div>"
                )
            elif ph == "break_long":
                phase_html = (
                    "<div style='background-color:rgba(255, 204, 204, 0.6); padding:4px; "
                    "border-radius:4px; margin-left:4px; margin-bottom:0;'>"
                    "<span style='font-size:16px; font-weight:600;'>💤 긴 휴식 시간</span>"
                    "</div>"
                )
            else:
                phase_html = ""
            phase_placeholder.markdown(phase_html, unsafe_allow_html=True)

            # ────────────────────────────── 3) break-short 파이 1개 ─────────────────────
            if ph == "break_short":
                # “짧은 휴식” 단계가 시작될 때만 파이를 그린다.
                if not st.session_state.break_short_drawn:
                    st.session_state.pie_placeholder.empty()

                    df_src = pd.DataFrame({"State": st.session_state.data["state"]})
                    counts = (
                        df_src["State"]
                        .value_counts()
                        .rename_axis("State")
                        .reset_index(name="Count")
                    )
                    counts["Ratio"] = counts["Count"] / counts["Count"].sum()

                    pie = (
                        alt.Chart(counts)
                        .mark_arc(innerRadius=50)
                        .encode(
                            theta="Count:Q",
                            color=alt.Color(
                                "State:N",
                                scale=alt.Scale(
                                    domain=["집중 상태","비집중 상태","외출 상태","졸음 상태"],
                                    range=["green","red","orange","blue"],
                                ),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("State:N", title="상태"),
                                alt.Tooltip("Ratio:Q", format=".1%", title="비율"),
                            ],
                        )
                        .properties(title="이전 뽀모도로 요약", width=200, height=200)
                    )
                    st.session_state.pie_placeholder.altair_chart(pie, use_container_width=True)

                    # 한 번 그렸음을 표시
                    st.session_state.break_short_drawn = True

            else:
                # “짧은 휴식” 단계가 끝난 순간 (직전 단계가 “break_short”였을 때)
                if ph != "break_short" and st.session_state.break_short_drawn:
                    st.session_state.break_short_drawn = False
                    st.session_state.pie_placeholder.empty()

            # ────────────────────────────── 4) break-long 파이 3개 ──────────────────────
            if ph == "break_long":
                # 긴 휴식 단계 첫 진입 시
                if not st.session_state.break_long_drawn:
                    st.session_state.pie_placeholder.empty()
                    draw_break_long_pies()
                    st.session_state.break_long_drawn = True
                else:
                    # 이미 그려져 있다면 캐시된 차트 객체를 다시 붙인다
                    charts = st.session_state.get("break_long_chart_objs")
                    if charts:
                        cols = st.session_state.pie_placeholder.columns(len(charts))
                        for col, ch in zip(cols, charts):
                            col.altair_chart(ch, use_container_width=True)

                # PNG 다운로드 버튼 처리
                if st.session_state.get("latest_focus_png") and not st.session_state.get("download_btn_drawn"):
                    png_path = st.session_state.latest_focus_png
                    if Path(png_path).exists():
                        with open(png_path, "rb") as f:
                            download_pl.download_button(
                                "PNG 다운로드", f,
                                file_name=Path(png_path).name,
                                key="download_png"
                            )
                        st.session_state.download_btn_drawn = True

            else:
                # “긴 휴식” 단계가 끝나면 캐시 초기화
                if st.session_state.last_displayed_phase == "break_long":
                    st.session_state.break_long_drawn = False
                    st.session_state.pie_placeholder.empty()


            # ────────────────────────────── 5) 타이머 표시 ──────────────────────────────
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
                    prev_phase = st.session_state.pomodoro_phase
                    transition_to_next_phase()
                    if (st.session_state.pomodoro_phase == "focus"
                        and prev_phase in ["break_short", "break_long"]):
                        st.session_state.state_events.clear()
                        st.session_state.log_placeholder.empty()
                        st.session_state.detect_prev_state = None
                        st.session_state.detect_start_time = None
                    st.session_state.prev_time_str = ""
            else:
                timer_placeholder.empty()
                st.session_state.prev_time_str = ""


            # ────────────────────────────── 6) 자동 스냅샷 ──────────────────────────────
            elapsed_snap = now - st.session_state.last_snapshot_time

            if elapsed_snap > 5:
                st.session_state.last_snapshot_time = now
                st.session_state.snapshot_display_until = now + 12  # 12초간 표시
                st.session_state.snap_img_elem.image(frame_rgb, use_container_width="auto")

            elif "snapshot_display_until" in st.session_state and now > st.session_state.snapshot_display_until:
                transparent_px = np.zeros((1, 1, 4), dtype=np.uint8)
                st.session_state.snap_img_elem.image(
                    transparent_px, channels="RGBA", use_container_width="auto"
                )
                st.session_state.pop("snapshot_display_until")


            # ────────────────────────────── 7) 외출/졸음 로그 출력 ──────────────────────────────
            if st.session_state.state_events:
                all_msgs = "\n\n".join(evt["message"] for evt in st.session_state.state_events)
                st.session_state.log_placeholder.markdown(all_msgs)


            # ────────────────────────────── 8) 상태 모니터링 차트 ──────────────────────────────
            df = pd.DataFrame({"Time": st.session_state.data["time"], "State": st.session_state.data["state"]})

            if not df.empty:
                tip = df["Time"].max()
                recent = df[df["Time"] >= tip - pd.Timedelta(seconds=30)].copy()
                recent["State_Visual"] = recent["State"].replace({"수면 상태": "졸음 상태"})

                chart = (
                    alt.Chart(recent)
                    .mark_circle(size=60)
                    .encode(
                        x="Time:T",
                        y="State:N",
                        color=alt.Color(
                            "State_Visual:N",
                            scale=alt.Scale(
                                domain=["집중 상태","비집중 상태","외출 상태","졸음 상태"],
                                range=["green","red","orange","blue"],
                            ),
                            legend=None,
                        ),
                        tooltip=["Time","State"],
                    )
                    .properties(height=150)
                )

                latest_ts = recent["Time"].iloc[-1]
                if latest_ts != st.session_state.get("last_chart_ts"):
                    st.session_state.chart_placeholder.altair_chart(chart, use_container_width=True)
                    st.session_state.last_chart_ts = latest_ts
            else:
                st.session_state.chart_placeholder.empty()

            # ────────────────────────────── 9) 직전 단계 저장 ──────────────────────────────
            st.session_state.last_displayed_phase = ph


            # 속도 유지 (약 5fps)
            time.sleep(0.2)

            # 루프 종료 조건
            if not st.session_state.running or not webrtc_ctx.state.playing:
                break

    elif not st.session_state.running:
        pass
    
if __name__ == '__main__':
    main()