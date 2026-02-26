# 🏥 CareMate (케어메이트)
> **AI 기반 만성질환 위험도 예측 및 멘탈 헬스 케어 솔루션**

**CareMate**는 사용자의 신체 정보, 생활 습관, 그리고 정신 건강(우울, 불안, 스트레스) 데이터를 종합적으로 분석하여 주요 만성질환(고혈압, 당뇨, 이상지질혈증, 뇌졸중)의 발병 위험도를 예측하는 AI 서비스입니다. 예측 결과에 따라 OpenAI 기반의 AI 상담사와 음성/텍스트로 심층 건강 상담을 진행할 수 있습니다.

---

## 🛠️ 기술 스택 (Tech Stack)

이 프로젝트는 다음과 같은 최신 데이터 사이언스 기술과 AI 모델링 기법을 사용하여 개발되었습니다.

### **Machine Learning & Data Science**
![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4C4B?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-3B4F63?style=flat&logo=lightgbm&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-blue?style=flat)

### **Application & AI Integration**
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=flat&logo=langchain&logoColor=white)

---

## 📊 머신러닝 모델링 (AI Modeling Process)

본 서비스의 핵심 예측 엔진은 **국민건강영양조사(KNHANES)** 데이터를 기반으로 학습되었습니다. 의료 데이터 특유의 **불균형(Imbalance)** 문제를 해결하고, 실제 환자를 놓치지 않기 위해 **재현율(Recall)** 최적화에 주력했습니다.

### 1. 데이터 전처리 및 특징 공학 (Preprocessing)
* **특성(Features):** 연령, 성별, 소득, BMI, 가족력(고혈압, 당뇨, 뇌졸중 등), 음주 여부, 수면 시간
* **정신건강 변수:** PHQ-9(우울), GAD-7(불안), BP1(스트레스), EQ-5D(삶의 질) 점수화 적용
* **스케일링:** `StandardScaler`를 사용하여 수치형 데이터 정규화

### 2. 불균형 데이터 처리 (Imbalance Handling)
* **SMOTE (Synthetic Minority Over-sampling Technique):** 환자 데이터가 적은 문제를 해결하기 위해 소수 클래스를 증강했습니다.
* **최적 비율:** 실험을 통해 `sampling_strategy=0.9` 비율에서 성능이 가장 우수함을 확인하고 적용했습니다.

### 3. 알고리즘 선정 및 최적화 (Optimization)
다양한 알고리즘(`Logistic Regression`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`)을 비교 분석하여 질환별 최적의 모델을 선정했습니다.

| 예측 질환 | 적용 모델 (Algorithm) | 선정 근거 |
|:---:|:---:|:---|
| **고혈압 (HTN)** | **Soft Voting Ensemble** <br> (LR + RF + GBM) | 단일 모델 대비 AUC 0.84로 일반화 성능 우수 |
| **이상지질혈증 (DY)** | **Soft Voting Ensemble** <br> (LR + RF + GBM) | 안정적인 예측 확률 분포 및 Recall 확보 |
| **당뇨 (DB)** | **Logistic Regression** | 데이터 과적합 방지 및 해석 용이성 확보 |
| **뇌졸중 (ST)** | **Logistic Regression** | 극심한 데이터 불균형 상황에서 가장 안정적인 성능 기록 |

* **하이퍼파라미터 튜닝:** `GridSearchCV` 및 `RandomizedSearchCV` 활용
* **임계값(Threshold) 조정:** `Precision-Recall Curve` 분석을 통해 Recall을 극대화하는 최적 임계값 적용

---

## 📱 서비스 주요 기능 (Key Features)

1. **사용자 건강 프로필 입력**
   - 기본 신체 정보(키, 체중), 사회경제적 지표, 가족력, 음주/수면 습관 입력
2. **심층 정신건강 설문 (Mental Health Survey)**
   - 의학적으로 검증된 4가지 설문 도구(PHQ-9, GAD-7, BP1, EQ-5D) 탑재
3. **AI 질환 예측 리포트**
   - 4대 만성질환 발병 확률 계산 및 **[정상/주의/위험]** 3단계 시각화
   - 우울, 불안, 스트레스 지수 시각화
4. **AI 음성 헬스케어 상담 (Health Assistant)**
   - 예측 결과(Context)를 인식한 **GPT-4o-mini** 기반 AI 상담사
   - **STT(음성 인식) & TTS(음성 합성)** 기술을 적용하여 어르신도 쉽게 대화 가능

---

## 🚀 실행 방법 (Installation & Usage)

### 1. 환경 설정 (Prerequisites)
* Python 3.8 이상이 설치되어 있어야 합니다.
* OpenAI API Key가 필요합니다.

### 2. 설치 및 실행 (Installation & Run)
```bash
# 1. 레포지토리 클론
git clone [https://github.com/3679gug/caremate.git](https://github.com/3679gug/caremate.git)
cd caremate

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 모델 파일 확인
# health_models.pkl 파일이 프로젝트 루트 경로에 있어야 합니다.
# 만약 파일이 없다면 health_models.zip 압축을 해제하세요.

# 4. 애플리케이션 실행
streamlit run caremate.py

# 5. 프로젝트 구조
caremate/
├── caremate.py             # 메인 Streamlit 애플리케이션 코드
├── health_models.pkl       # 학습 완료된 AI 모델 (Ensemble & Logistic)
├── health_models.zip       # 모델 백업 파일 (압축)
├── 모델 학습(0208).ipynb    # 모델 학습, 튜닝, 평가 과정 노트북
├── logo.gif                # 앱 로고 리소스
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # 프로젝트 설명서
