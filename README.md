# Pomodoro_with_state_tracking

### 레포지터리
<pre><code>
Pomodoro_with_state_tracking
├── demo  # 상태 추적 데모  
│   ├── drowsy_estimation_with_FaceMesh.py
│   ├── gaze_estimation_with_FaceMesh_OpenVINO.py
│   ├── gaze_estimation_with_OpenVINO.py
│   ├── landmarks_visualization_with_OpenVINO.py
│   └── sleeping_outing_estimation_with_Pose.py
├── OpenVINO
│   ├── face-detection-adas-0001
│   │   └── FP32
│   │       └── face-detection-adas-0001  # 얼굴 감지 모델  
│   ├── head-pose-estimation-adas-0001
│   │   └── FP32
│   │       └── head-pose-estimation-adas-0001  # 머리 방향 추출 모델  
│   └── gaze-estimation-adas-0002
│       └── FP32
│           └── gaze-estimation-adas-0002  # 시선 벡터 추출 모델
├── config_model.py   # 모델 로드
├── drowsy_estimation.py   # 졸음 판단 로직
├── gaze_estimation.py  # 집중 판단 로직 
├── main.py  # 메인 실행
├── main_base.py  # 뽀모도로 없는 버전 
├── pomodoro_timer.py  # 뽀모도로 타이머 로직 
└── requirement.txt
</code></pre>


### 실행
<pre><code>
streamlit run main.py
</code></pre>
