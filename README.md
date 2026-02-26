#🛡️ 케어메이트 (CareMate) - AI 만성질환 예측 및 상담 서비스
케어메이트는 사용자의 건강 데이터와 정신건강 설문 결과를 결합하여 고혈압, 당뇨 등 주요 만성질환의 발병 위험도를 예측하고, AI 챗봇을 통해 개인 맞춤형 건강 상담을 제공하는 Streamlit 기반 웹 애플리케이션입니다.

🚀 주요 기능 (Key Features)
1. 개인 건강 정보 관리
사용자 인증: 회원가입, 로그인 및 비회원 시작 모드 지원.

프로필 입력: 성별, 나이, 신체 정보(BMI 자동 계산), 교육 수준, 소득 수준, 음주 습관, 수면 시간 등 상세 건강 데이터 수집.

가족력 기록: 고혈압, 당뇨병, 뇌졸중 등 유전적 요인 반영.

2. 정신건강 및 삶의 질 설문
PHQ-9: 우울증 선별 검사.

GAD-7: 불안 장애 선별 검사.

BP1: 일상적 스트레스 인지 수준 측정.

EQ-5D: 건강 관련 삶의 질 측정 (가중치 기반 지수 계산).

3. AI 만성질환 예측 리포트
머신러닝 예측: 저장된 모델(health_models.pkl)을 통해 고혈압, 이상지질혈증, 당뇨, 뇌졸중의 위험도 분석.

위험도 등급화: 임계값(Threshold)을 기준으로 '높음', '중간', '낮음' 3단계 시각화 제공.

종합 의견: 분석 결과를 바탕으로 전문의 상담 권고 또는 생활습관 개선 가이드 제시.

4. AI 음성 챗봇 상담
멀티모달 대화: 텍스트 입력뿐만 아니라 **STT(Speech-to-Text)**를 통한 음성 인식 지원.

맞춤형 상담: 사용자의 건강 프로필과 예측 결과를 문맥으로 이해하는 GPT-4o-mini 기반 상담.

음성 답변 (TTS): gTTS를 활용하여 AI의 답변을 음성으로 출력.

🛠 기술 스택 (Tech Stack)
프레임워크 및 UI
Streamlit: 웹 인터페이스 구성 및 상태 관리.

CSS Custom Styling: 사용자 경험 최적화를 위한 카드형 레이아웃 및 디자인 적용.

데이터 분석 및 머신러닝
Pandas & NumPy: 데이터 전처리 및 수치 계산.

Scikit-learn: 예측 모델(Pipeline, Logistic Regression, Ensemble) 활용.

Pickle: 학습된 모델 로드.

AI 및 언어 모델
LangChain: OpenAI LLM과의 효율적인 상호작용 및 프롬프트 엔지니어링.

OpenAI GPT-4o-mini: 지능형 건강 상담 엔진.

gTTS (Google Text-to-Speech): 결과 음성 변환.

Streamlit Mic Recorder: 사용자 음성 입력 처리.

⚙️ 설치 및 실행 방법 (Installation)
저장소 클론

Bash
git clone https://github.com/your-username/caremate.git
cd caremate
필수 라이브러리 설치

Bash
pip install streamlit pandas numpy langchain-openai gtts streamlit-mic-recorder scikit-learn
모델 파일 준비

health_models.zip 또는 health_models.pkl 파일이 루트 디렉토리에 있어야 합니다. (실행 시 자동으로 압축 해제 및 로드 프로세스 진행)

API 키 설정

코드 내 OPENAI_API_KEY를 본인의 키로 교체하거나 Streamlit Secrets를 사용하세요.

실행

Bash
streamlit run caremate.py
📊 데이터 모델 정보
본 서비스는 국민건강영양조사(KNHANES) 데이터를 기반으로 학습된 모델을 사용하여 임상적 유의성을 높였습니다.

⚠️ 유의사항
본 서비스에서 제공하는 예측 결과는 의학적 진단을 대신할 수 없습니다. 정확한 상태 확인을 위해서는 반드시 의료기관을 방문하시기 바랍니다.
