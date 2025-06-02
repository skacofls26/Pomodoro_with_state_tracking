'''
pomodoro.py
    뽀모도로 타이머 로직
'''
import streamlit as st
import pandas as pd
import altair as alt
from plyer import notification
from datetime import datetime
import time

# ========================= 1) 단계별 지속시간 반환 =========================
def get_current_phase_duration():
    """
    현재 단계(집중/휴식)에서 남은 총 지속시간(초)을 반환합니다.
    """
    p = st.session_state.pomodoro_phase
    if p == "focus":
        return st.session_state.pomodoro_duration_focus
    if p == "break_short":
        return st.session_state.pomodoro_duration_break
    if p == "break_long":
        return st.session_state.break_long_duration
    return 0

# ========================= 3) 단계 전환 및 알림 =========================
def transition_to_next_phase():
    """
    현재 단계가 종료되었을 때 호출됩니다.
    - 집중 종료 시: 알림, 휴식 시작 시간 기록, 다음을 짧은 휴식 또는 긴 휴식으로 전환
    - 휴식 종료 시: 세션 저장&시각화, 알림, 다음 집중 단계로 전환
    """
    p = st.session_state.pomodoro_phase
    c = st.session_state.cycle_count

    if p == "focus":
        # ─ 집중 단계가 끝나는 순간: pie chart 그리기 ─

        # 1) 현재까지 수집된 상태 데이터 집계
        df_focus = pd.DataFrame({
            "Time": st.session_state.data["time"],
            "State": st.session_state.data["state"]
        })
        # 상태별 카운트 계산
        counts = df_focus["State"].value_counts().reset_index()
        counts.columns = ["State", "Count"]
        counts["Percent"] = counts["Count"] / counts["Count"].sum()

        # ─ 기존 집중 종료 알림 ─
        notification.notify(
            title="뽀모도로 알림",
            message=f"{c}번째 집중이 끝났습니다. 잠시 휴식을 취하세요!",
            timeout=5,
        )

        # 4) 휴식 시작 시각 저장
        st.session_state.break_start_dt = datetime.now()

        # 5) 단계 전환: short/long break
        if c < 4:
            st.session_state.pomodoro_phase = "break_short"
        else:
            st.session_state.pomodoro_phase = "break_long"
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start = time.monotonic()

    elif p == "break_short":
        # 휴식이 끝나면 저장

        notification.notify(
            title="뽀모도로 알림",
            message="짧은 휴식이 끝났습니다. 다시 집중을 시작하세요!",
            timeout=5,
        )

        st.session_state.cycle_count += 1
        st.session_state.pomodoro_phase = "focus"
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start = time.monotonic()
        st.session_state.session_count += 1

    elif p == "break_long":

        notification.notify(
            title="뽀모도로 알림",
            message="긴 휴식이 끝났습니다. 새로운 사이클을 시작하세요!",
            timeout=5,
        )

        st.session_state.cycle_count = 1
        st.session_state.pomodoro_phase = "focus"
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start = time.monotonic()
        st.session_state.session_count = 1