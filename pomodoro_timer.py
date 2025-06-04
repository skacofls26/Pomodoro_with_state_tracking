'''
pomodoro_timer.py
    뽀모도로 타이머 로직
'''
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
from plyer import notification
import time
import altair as alt

DATA_DIR = Path("/mnt/data")

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame(Column0, Column1 …) → 반드시
    'State', 'Count' 두 컬럼을 갖도록 변환
    """
    if "Count" in df.columns and "State" in df.columns:
        return df

    cols = list(df.columns)
    # 첫 컬럼을 State, 두 번째를 Count 로 간주
    new_df = df.copy()
    new_df.columns = ["State", "Count"] + cols[2:]
    return new_df[["State", "Count"]]


def save_focus_pies_png(buffer: list[pd.DataFrame]) -> Path:
    """
    최근 3회 집중 통계를 pie 3개로 그리고
    /mnt/data/YYMMDD_HHMM_summary.png 에 저장.
    반환값: Path 객체
    """
    if not buffer:
        return None

    # ---------- 스타일 ----------
    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 150,
    })

    # ----------- 준비 -----------
    n = len(buffer)
    fig, axes = plt.subplots(
        1, n,
        figsize=(4 * n, 6),            # 가로: 4*n, 세로: 6
        constrained_layout=False
    )
    axes = [axes] if n == 1 else axes

    # 색상·라벨 매핑 (영어 키 기준)
    color_map = {
        "Focused":   "#4CAF50",  # 녹색
        "Unfocused": "#F44336",  # 빨간
        "Away":      "#FFC107",  # 노란
        "Drowsy":    "#2196F3",  # 파란
    }
    order = list(color_map.keys())

    # 한글→영어 매핑 (DF의 State가 한글일 경우)
    kor2eng = {
        "집중 상태":   "Focused",
        "비집중 상태": "Unfocused",
        "외출 상태":   "Away",
        "졸음 상태":   "Drowsy",
    }

    # ----------- 각 Pie -----------
    for ax, (idx, item) in zip(axes, enumerate(buffer, 1)):
        df, dur = item  # df: DataFrame, dur: {"F":25,"S":5,"L":15}

        # 1) State 컬럼을 영어로 바꿔주기
        df = df.replace({"State": kor2eng})

        # 2) 순서를 보장하면서 index 세팅
        df = df.set_index("State").reindex(order, fill_value=0).reset_index()

        # 3) 파이 그리기 (autopct 제거)
        ax.pie(
            df["Count"],
            labels=None,             # 파이 위의 레이블도 표시하지 않음
            pctdistance=0.72,
            labeldistance=1.10,
            colors=[color_map[s] for s in df["State"]],
            startangle=90,
            wedgeprops=dict(width=0.35, edgecolor="white"),
            textprops={"fontsize": 8},
        )

        # 4) Pie 개별 캡션 (위쪽)
        cap = f'{dur["F"]}m {dur["S"]}m {dur["L"]}m'
        ax.set_title(
            f"Pomodoro {idx}\n{cap}",
            fontsize=13,               # 제목 폰트 크기
            fontweight='bold',
            pad=4                      # 제목과 그래프 간격을 더 좁게
        )

        # 5) Pie 아래에 “● 레이블: 백분율” 형태로 출력하기
        total = df["Count"].sum() or 1  # 0으로 나누기 방지
        # 상대 좌표(ax.transAxes) 사용. y_start는 파이 바로 아래.
        y_start = -0.02                 # 파이 바로 아래에서 시작 (더 가깝게)
        y_step  = 0.05                  # 한 줄당 내려가는 거리 (4줄 간격 적절히)

        for i, state in enumerate(order):
            count = int(df.loc[df["State"] == state, "Count"].item())
            pct = (count / total) * 100

            # 5-1) ● 기호만 컬러로 출력 (x=0.00)
            ax.text(
                0.00,                            # 기호 x 좌표
                y_start - i * y_step,            # y 좌표
                "\u25CF",                        # Unicode ● 기호
                color=color_map[state],          # 기호 색깔
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )
            # 5-2) 기호 옆에 레이블 텍스트를 검정색으로 출력 (x=0.08)
            ax.text(
                0.08,                                         # ● 기호보다 더 오른쪽
                y_start - i * y_step,                         # 같은 y 좌표
                f"{state}: {pct:.0f}%",                       # 레이블 + 백분율
                color="black",                                # 레이블 텍스트는 검정
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )

        # 6) Pie 외곽을 표시하기 위해 축 비활성
        ax.axis("equal")   # 원이 찌그러지지 않게
        ax.set_xticks([])
        ax.set_yticks([])

    # ----------- 레이아웃 & 저장 -----------
    # 전체 바깥 여백을 넉넉히 확보 (rect를 전체로, pad에 큰 값 지정)
    fig.tight_layout(pad=5.0, w_pad=5.0, rect=[0.00, 0.00, 1.00, 1.00])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M")
    path = DATA_DIR / f"{ts}_summary.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def draw_break_long_pies():
    buf = st.session_state.get("focus_counts_buffer", [])
    n   = len(buf)
    if not n:
        return

    cols   = st.session_state.pie_placeholder.columns(n)
    domain = ["집중 상태", "비집중 상태", "외출 상태", "졸음 상태"]
    colors = ["#4CAF50", "#F44336", "#FFC107", "#2196F3"]
    chart_objs = []

    for col, (idx, (df_one, dur)) in zip(cols, enumerate(buf, 1)):
        counts = (
            df_one.set_index("State")
                  .reindex(domain, fill_value=0)
                  .reset_index()
        )
        total = counts["Count"].sum() or 1
        counts["Ratio"] = counts["Count"] / total

        pie = (
            alt.Chart(counts)
               .mark_arc(innerRadius=50)
               .encode(
                   theta="Count:Q",
                   color=alt.Color("State:N",
                                   scale=alt.Scale(domain=domain, range=colors),
                                   legend=None),
                   tooltip=[alt.Tooltip("State:N", title="상태"),
                            alt.Tooltip("Ratio:Q", format=".1%", title="비율")],
               )
               .properties(title=f"{idx}번째 뽀모도로 요약",
                           width=220, height=220)
        )
        col.altair_chart(pie, use_container_width=True)
        chart_objs.append(pie)

    # 🔑 캐시 : 차트와 column 둘 다
    st.session_state.break_long_chart_objs = chart_objs
    st.session_state.break_long_cols       = cols
    st.session_state.break_long_buf_len    = n

# --------------------------------------------------------------

# ========================= 1) 단계별 지속시간 반환 =========================
def get_current_phase_duration():
    phase = st.session_state.pomodoro_phase
    if phase == "focus":
        return st.session_state.pomodoro_duration_focus
    if phase == "break_short":
        return st.session_state.pomodoro_duration_break
    if phase == "break_long":
        return st.session_state.break_long_duration
    return 0

# ========================= 2) 단계 전환 =========================
def transition_to_next_phase():
    """
    타이머가 0이 될 때 호출.
    - 집중 → (3회마다 긴, 나머지 짧은) 휴식
    - 짧은 휴식 → 다음 집중
    - 긴 휴식 → 바로 다음 집중(자동 재시작)
    """
    phase  = st.session_state.pomodoro_phase
    cycle  = st.session_state.cycle_count

    # ───────── ① 집중 종료 ─────────
    if phase == "focus":
        # 이번 집중 통계 만들기
        df_now = (
            pd.Series(st.session_state.data["state"])
            .value_counts()
            .reset_index()
            .rename(columns={"index": "State", 0: "Count"})
        )
        df_now = _standardize(df_now)  
        st.session_state.break_short_snapshot = df_now 

        dur = {
            "F": int(st.session_state.pomodoro_duration_focus  // 60),
            "S": int(st.session_state.pomodoro_duration_break  // 60),
            "L": int(st.session_state.break_long_duration      // 60),
        }

        # 버퍼(3개 유지)
        buf = st.session_state.get("focus_counts_buffer", [])
        buf.append((df_now, dur))
        st.session_state.focus_counts_buffer = buf[-3:]

        # ── PNG 저장
        if cycle % 3 == 0 and len(st.session_state.focus_counts_buffer) == 3:
            png_path = save_focus_pies_png(st.session_state.focus_counts_buffer)
            st.session_state.latest_focus_png = str(png_path)
        else:
            # 이전 PNG 값 제거 (break-short 에서 안 보이도록)
            st.session_state.pop("latest_focus_png", None)

        # ▼ 3) 알림
        notification.notify(
            title="뽀모도로 알림",
            message=f"{cycle}번째 집중이 끝났습니다. 휴식을 취하세요!",
            timeout=5,
        )

        # ▼ 4) 휴식 종류 결정
        if cycle % 3 == 0:          # 3회째마다 긴 휴식
            # 직전 3회의 합산 통계 → long_break_counts
            st.session_state.pomodoro_phase = "break_long"
        else:
            st.session_state.pomodoro_phase = "break_short"

        # ▼ 5) 휴식 타이머 시작
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start   = time.monotonic()
        st.session_state.break_start_dt   = datetime.now()

    # ───────── ② 짧은 휴식 종료 ─────────
    elif phase == "break_short":
        notification.notify(
            title="뽀모도로 알림",
            message=f"짧은 휴식이 끝났습니다. {cycle+1}번째 집중을 시작하세요!",
            timeout=5,
        )
        st.session_state.cycle_count     += 1
        st.session_state.session_count   += 1
        st.session_state.pomodoro_phase   = "focus"
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start   = time.monotonic()

    # ───────── ③ 긴 휴식 종료 ─────────
    elif phase == "break_long":
        notification.notify(
            title="뽀모도로 알림",
            message=f"긴 휴식이 끝났습니다. {cycle+1}번째 집중을 시작합니다!",
            timeout=5,
        )
        # 새 세트로 카운터 초기화하면서 **자동 재시작**
        st.session_state.cycle_count     += 1
        st.session_state.session_count   += 1
        st.session_state.pomodoro_phase  = "focus"
        st.session_state.pomodoro_elapsed = 0.0
        st.session_state.pomodoro_start   = time.monotonic()
        # 버퍼 리셋
        st.session_state.focus_counts_buffer = []

        st.session_state.last_displayed_cycle  = None
        st.session_state.last_displayed_phase  = None
        st.session_state.pie_placeholder.empty()
