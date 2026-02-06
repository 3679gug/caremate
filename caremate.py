import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import io
import base64
import zipfile # 압축 해제를 위해 추가

# --- AI 및 음성 기능을 위한 라이브러리 추가 ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# [보안 설정 영역]
# ---------------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# OPENAI_API_KEY = ""
# --- 디자인 설정 ---
STYLE_CONFIG = {
 "corner_radius": "25px",      
 "border_width": "1px",
 "border_color": "#e2e8f0",
 "fg_color": "#FFFFFF",
 "bg_color": "#F0F9F6",
 "primary_color": "#ff4b4b"
}

LEVEL_THEMES = {
 "높음": {"color": "#ef4444", "bg": "#fee2e2", "emoji": "🔴"},
 "중간": {"color": "#f59e0b", "bg": "#fef3c7", "emoji": "🟡"},
 "낮음": {"color": "#22c55e", "bg": "#dcfce7", "emoji": "🟢"}  
}

st.set_page_config(page_title="케어메이트 - AI 만성질환 예측", layout="centered", page_icon="🏥")

# --- 이미지 로드 함수 (Base64 변환) ---
def get_image_base64(path):
 if os.path.exists(path):
  with open(path, "rb") as img_file:
   return base64.b64encode(img_file.read()).decode()
 return None

# --- [수정 부분] ZIP 압축 해제 로직 포함 모델 로드 ---
@st.cache_resource
def load_models():
 zip_path = 'health_models.zip'
 model_path = 'health_models.pkl'
 
 # 1. 만약 pkl 파일이 없고 zip 파일만 있다면 압축 해제 실행
 if not os.path.exists(model_path):
  if os.path.exists(zip_path):
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('.')
  else:
   st.error("❌ 모델 파일(health_models.pkl 또는 health_models.zip)이 없습니다.")
   st.stop()
 
 # 2. 모델 로드 및 AttributeError 방어 로직
 with open(model_path, 'rb') as f:
  models_package = pickle.load(f)
 
 # LogisticRegression의 multi_class 속성 누락 오류 해결을 위한 패치
 for target, pipeline in models_package['pipelines'].items():
  # 파이프라인의 마지막 단계(모델) 추출
  final_estimator = pipeline.steps[-1][1]
  
  # 1. 만약 보팅 앙상블 모델인 경우 내부의 모든 모델 검사
  if hasattr(final_estimator, 'estimators_'):
   for est in final_estimator.estimators_:
    # 개별 estimator가 파이프라인일 경우 그 안의 실제 모델 추출
    actual_model = est.steps[-1][1] if hasattr(est, 'steps') else est
    if 'LogisticRegression' in str(type(actual_model)) and not hasattr(actual_model, 'multi_class'):
     actual_model.multi_class = 'ovr' # 또는 'auto'
  
  # 2. 단일 로지스틱 회귀 모델인 경우
  elif 'LogisticRegression' in str(type(final_estimator)) and not hasattr(final_estimator, 'multi_class'):
   final_estimator.multi_class = 'ovr'
   
 return models_package

MODELS_DATA = load_models()

# --- 세션 상태 초기화 ---
if 'db' not in st.session_state: st.session_state.db = {} 
if 'step' not in st.session_state: st.session_state.step = 0 
if 'auth_mode' not in st.session_state: st.session_state.auth_mode = "main"
if 'current_user' not in st.session_state: st.session_state.current_user = None
if 'is_existing_user' not in st.session_state: st.session_state.is_existing_user = False
if 'sub_step' not in st.session_state: st.session_state.sub_step = 1
if 'q_idx' not in st.session_state: st.session_state.q_idx = 0
if 'data_confirmed' not in st.session_state: st.session_state.data_confirmed = False
if 'user_data' not in st.session_state:
 st.session_state.user_data = {
  "name": "", "gender": "남성", "age": 70, "height": 160, "weight": 60,
  "diseases": [], "family_history": [], "edu": "대졸 이상", "marry": "기혼",
  "incm": "상(500만원~)", "alcohol": "비음주", "sleep_time": 7
 }
if 'survey_answers' not in st.session_state:
 st.session_state.survey_answers = {"PHQ9": {}, "GAD7": {}, "BP1": {}, "EQ5D": {}}
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'tts_enabled' not in st.session_state: st.session_state.tts_enabled = True 

# --- CSS 스타일 ---
st.markdown(f"""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
 .stApp {{ background-color: {STYLE_CONFIG['bg_color']} !important; font-family: 'Noto Sans KR', sans-serif; }}
 .block-container {{ max-width: 700px !important; padding: 3rem 1rem !important; }}
 
 [data-testid="stVerticalBlock"] > div:has(div.card-content) {{
  background-color: white !important; padding: 40px !important;
  border-radius: {STYLE_CONFIG['corner_radius']} !important;
  border: {STYLE_CONFIG['border_width']} solid {STYLE_CONFIG['border_color']} !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
 }}
 
 .summary-box {{
  background-color: #f8fafc; padding: 20px; border-radius: 15px; border: 1px solid #e2e8f0; margin: 20px 0;
 }}
 
 .disease-item-card {{ background-color: white; border-radius: 18px; padding: 22px; margin-bottom: 15px; border: 1px solid #edf2f7; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
 .chat-bubble-ai {{ background-color: #f1f5f9; padding: 12px; border-radius: 15px; margin-bottom: 10px; color: #334155; }}
 .chat-bubble-user {{ background-color: #22c55e; padding: 12px; border-radius: 15px; margin-bottom: 10px; color: white; text-align: right; }}
 
 div[role="radiogroup"] {{ display: flex; flex-direction: column; gap: 12px !important; }}
 div[role="radiogroup"] > label {{ 
  background-color: white !important; border: 1px solid #e2e8f0 !important; 
  border-radius: 12px !important; padding: 12px 15px !important; 
  width: 100% !important; display: flex !important; margin-bottom: 0px !important;
 }}
 
 div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {{
  flex-direction: row !important;
  gap: 20px !important;
 }}
 div[data-testid="stHorizontalBlock"] div[role="radiogroup"] > label {{
  width: auto !important;
  flex: 1 !important;
 }}

 div[role="radiogroup"] > label[data-checked="true"] {{ border-color: {STYLE_CONFIG['primary_color']} !important; background-color: #fffafa !important; }}
 div[role="radiogroup"] > label[data-checked="true"] p {{ color: {STYLE_CONFIG['primary_color']} !important; font-weight: 600 !important; }}
 
 button[kind="primary"] {{ background-color: {STYLE_CONFIG['primary_color']} !important; border: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- 계산 및 예측 함수 ---
def calculate_scores():
 phq_m = {"전혀 아니다": 0, "여러 날 동안": 1, "일주일 이상": 2, "거의 매일": 3}
 phq = sum([phq_m.get(next((k for k in phq_m if k in v), ""), 0) for v in st.session_state.survey_answers['PHQ9'].values()])
 gad_m = {"전혀 아니다": 0, "며칠 동안": 1, "7일 이상": 2, "거의 매일": 3}
 gad = sum([gad_m.get(next((k for k in gad_m if k in v), ""), 0) for v in st.session_state.survey_answers['GAD7'].values()])
 bp1_score = 1
 if st.session_state.survey_answers['BP1']:
  ans = list(st.session_state.survey_answers['BP1'].values())[0]
  bp1_score = int(ans.split(".")[0]) if "." in ans else 1
 eq_ans = [int(v.split(".")[0]) if v and "." in v else 1 for v in st.session_state.survey_answers['EQ5D'].values()]
 while len(eq_ans) < 5: eq_ans.append(1)
 m2, m3 = (1, 0) if eq_ans[0]==2 else (0, 1) if eq_ans[0]==3 else (0, 0)
 sc2, sc3 = (1, 0) if eq_ans[1]==2 else (0, 1) if eq_ans[1]==3 else (0, 0)
 ua2, ua3 = (1, 0) if eq_ans[2]==2 else (0, 1) if eq_ans[2]==3 else (0, 0)
 pd2, pd3 = (1, 0) if eq_ans[3]==2 else (0, 1) if eq_ans[3]==3 else (0, 0)
 ad2, ad3 = (1, 0) if eq_ans[4]==2 else (0, 1) if eq_ans[4]==3 else (0, 0)
 n3 = 1 if 3 in eq_ans else 0
 eq5d = 1 - (0.05 + 0.096*m2 + 0.418*m3 + 0.046*sc2 + 0.209*sc3 + 0.038*ua2 + 0.192*ua3 + 0.058*pd2 + 0.278*pd3 + 0.062*ad2 + 0.19*ad3 + 0.05*n3)
 return phq, gad, bp1_score, eq5d

def get_predictions():
 u = st.session_state.user_data
 bmi = u['weight'] / ((u['height']/100)**2)
 phq, gad, bp1, eq5d = calculate_scores()
 
 alc_map = {"비음주": 0, "적정 음주": 1, "고위험 음주": 2}
 edu_map = {"초졸 이하": 1, "중졸": 2, "고졸": 3, "대졸 이상": 4}
 marry_map = {"기혼": 1, "미혼": 2, "이혼/사별/기타": 3}
 incm_map = {"하(~244 만원)": 1, "중하(244~356 만원)": 2, "중상(244~356 만원)": 3, "상(500만원~)": 4}
 
 full_data = {
  'age': u['age'], 
  'sex': 1 if u['gender'] == "남성" else 2, 
  'edu': edu_map.get(u['edu'], 3), 
  'marry': marry_map.get(u['marry'], 1), 
  'FH_HE': 1 if "고혈압" in u['family_history'] else 0, 
  'FH_DB': 1 if "당뇨병" in u['family_history'] else 0, 
  'FH_DY': 1 if "이상지질혈증" in u['family_history'] else 0, 
  'FH_HAA': 1 if "뇌졸중" in u['family_history'] else 0, 
  'HE_BMI': bmi, 
  'alcohol': alc_map.get(u['alcohol'], 0), 
  'mh_PHQ_S': phq, 
  'mh_GAD_S': gad, 
  'BP1': bp1, 
  'EQ5D': eq5d, 
  'sleep_time_wy': u['sleep_time'], 
  'incm': incm_map.get(u['incm'], 4)
 }
 
 predictions = {}
 for target, pipeline in MODELS_DATA['pipelines'].items():
  features = MODELS_DATA['features'][target]
  threshold = MODELS_DATA['thresholds'][target]
  input_df = pd.DataFrame([[full_data.get(f, 0) for f in features]], columns=features)
  
  # 예측 수행
  prob = pipeline.predict_proba(input_df)[0, 1]
  name_kr = {"clinical_HE": "고혈압", "clinical_DY": "이상지질혈증", "clinical_DB": "당뇨", "clinical_ST": "뇌졸중"}.get(target, target)
  predictions[name_kr] = {"prob": prob, "threshold": threshold}
 return predictions

# --- STEP 0: 메인화면 ---
if st.session_state.step == 0:
 with st.container():
  st.markdown('<div class="card-content" style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">', unsafe_allow_html=True)
  logo_base64 = get_image_base64("logo.gif") 
  if logo_base64:
   st.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="400" style="margin-bottom:10px; display: block; margin-left: auto; margin-right: auto;">', unsafe_allow_html=True)
  else:
   st.markdown('<div style="width: 100%; text-align: center;"><h1 style="font-size: 4rem; margin-bottom:10px;">🛡️</h1></div>', unsafe_allow_html=True)
  st.markdown('<h1 style="width: 100%; text-align: center; margin-bottom:40px; font-weight: 700;">케어메이트</h1>', unsafe_allow_html=True)
  
  if st.session_state.auth_mode == "main":
   if st.button("📲 기존 계정으로 로그인", type="primary", use_container_width=True): 
    st.session_state.auth_mode = "login"; st.rerun()
   st.write("")
   if st.button("👤 새 회원가입", use_container_width=True): 
    st.session_state.auth_mode = "signup"; st.rerun()
   st.markdown("<br>", unsafe_allow_html=True)
   if st.button("🔒 비회원으로 시작하기", use_container_width=True): 
    st.session_state.is_existing_user = False
    st.session_state.step = 1; st.rerun()
  elif st.session_state.auth_mode == "login":
   l_id = st.text_input("아이디")
   l_pw = st.text_input("비밀번호", type="password")
   if st.button("로그인", type="primary", use_container_width=True):
    if l_id in st.session_state.db and st.session_state.db[l_id]['pw'] == l_pw:
     st.session_state.current_user = l_id; 
     st.session_state.user_data = st.session_state.db[l_id]['data']
     st.session_state.is_existing_user = True
     st.session_state.step = 1; st.rerun() 
    else: st.error("정보가 일치하지 않습니다.")
   if st.button("뒤로가기"): st.session_state.auth_mode = "main"; st.rerun()
  elif st.session_state.auth_mode == "signup":
   n_id = st.text_input("사용할 아이디")
   n_pw = st.text_input("사용할 비밀번호", type="password")
   if st.button("가입하기", type="primary", use_container_width=True):
    if n_id and n_pw:
     if n_id in st.session_state.db:
      st.error(f"❌ '{n_id}'은(는) 이미 사용 중인 아이디입니다. 다른 아이디를 입력해 주세요.")
     else:
      st.session_state.db[n_id] = {'pw': n_pw, 'data': st.session_state.user_data.copy()}
      st.session_state.current_user = n_id
      st.session_state.is_existing_user = False
      st.success(f"✅ {n_id}님, 회원가입이 완료되었습니다!")
      st.session_state.step = 1; st.rerun()
    else: st.warning("아이디와 비밀번호를 모두 입력하세요.")
   if st.button("뒤로가기"): st.session_state.auth_mode = "main"; st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 1: 건강 정보 입력 ---
elif st.session_state.step == 1:
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  
  if st.session_state.is_existing_user:
   title_text = "🏥 개인 건강 정보 입력-수정"
  else:
   title_text = "🏥 개인 건강 정보 입력"
  st.markdown(f'<h2 style="text-align:center; margin-bottom:30px;">{title_text}</h2>', unsafe_allow_html=True)
  
  c1, c2 = st.columns(2)
  with c1: 
   name = st.text_input("성함", value=st.session_state.user_data["name"])
   if not name:
    st.markdown("<p style='color:red; font-size:0.8rem; margin-top:-15px;'>성함을 입력하세요</p>", unsafe_allow_html=True)
  with c2: 
   gender = st.radio("성별", ["남성", "여성"], index=0 if st.session_state.user_data["gender"]=="남성" else 1, horizontal=True)
  
  c3, c4 = st.columns(2)
  with c3: edu = st.selectbox("교육 수준", ["초졸 이하", "중졸", "고졸", "대졸 이상"], index={"초졸 이하":0, "중졸":1, "고졸":2, "대졸 이상":3}.get(st.session_state.user_data["edu"], 3))
  with c4: marry = st.selectbox("결혼 여부", ["기혼", "미혼", "이혼/사별/기타"], index={"기혼":0, "미혼":1, "이혼/사별/기타":2}.get(st.session_state.user_data["marry"], 0))
  
  st.divider()
  col_a, col_b, col_c = st.columns(3)
  with col_a: age = st.number_input("나이 (세)", 1, 120, st.session_state.user_data["age"])
  with col_b: height = st.number_input("키 (cm)", 50, 250, st.session_state.user_data["height"])
  with col_c: weight = st.number_input("몸무게 (kg)", 20, 200, st.session_state.user_data["weight"])
  
  col_d, col_e, col_f = st.columns(3)
  with col_d: incm = st.selectbox("소득 수준(월소득 기준)", ["하(~244 만원)", "중하(244~356 만원)", "중상(356~500 만원)", "상(500만원~)"], index={"하(1분위)":0, "중하(2분위)":1, "중상(3분위)":2, "상(4분위)":3}.get(st.session_state.user_data["incm"], 3))
  with col_e: 
   alc_guide = "7잔" if gender == "남성" else "5잔"
   alcohol = st.radio("음주 습관", ["비음주", "적정 음주", "고위험 음주"], index=["비음주", "적정 음주", "고위험 음주"].index(st.session_state.user_data.get("alcohol", "비음주")), horizontal=True)
   st.caption(f"※ 고위험: 월 1회 평균 {alc_guide} 이상 음주")
  with col_f: sleep = st.number_input("평균 수면시간", 0, 24, st.session_state.user_data["sleep_time"])
  
  st.divider()
  family_history = st.multiselect("가족력", ["고혈압", "당뇨병", "이상지질혈증", "심근경색 및 협심증", "뇌졸중","없음"], default=st.session_state.user_data["family_history"])
  
  updated_data = {"name": name, "gender": gender, "age": age, "height": height, "weight": weight,  "family_history": family_history, "edu": edu, "marry": marry, "incm": incm, "alcohol": alcohol, "sleep_time": sleep}
  st.session_state.user_data.update(updated_data)
  
  if st.session_state.current_user:
   st.session_state.db[st.session_state.current_user]['data'] = st.session_state.user_data.copy()

  st.markdown(f"""
   <div class="summary-box">
    <p style="margin:0; font-weight:700; color:{STYLE_CONFIG['primary_color']}; font-size:1.1rem;">📋 입력 정보 요약 확인</p>
    <p style="margin:8px 0 0 0; font-size:1rem; line-height:1.6;">
     성함: <b>{name if name else "___"}</b> 님 ({gender}) | 나이: <b>{age}세</b><br>
     신체: <b>{height}cm / {weight}kg</b> | 수면: <b>{sleep}시간</b><br>
     학력: <b>{edu}</b> | 결혼: <b>{marry}</b><br>
     소득: <b>{incm}</b> | 음주: <b>{alcohol}</b><b> | 가족력: <b>{", ".join(family_history) if family_history else "없음"}</b>
    </p>
   </div>
  """, unsafe_allow_html=True)
  
  st.write("위 정보가 정확합니까?")
  conf_col1, conf_col2 = st.columns(2)
  with conf_col1:
   if st.button("네, 맞습니다 ➡", type="secondary", use_container_width=True):
    if not name: st.error("성함을 입력해 주세요.")
    else: st.session_state.data_confirmed = True; st.rerun()
  with conf_col2:
   if st.button("아니오, 수정하겠습니다", use_container_width=True):
    st.session_state.data_confirmed = False; st.info("상단 입력란에서 내용을 수정해 주세요.")

  if st.session_state.data_confirmed:
   if st.button("정신건강 설문 시작하기 ➡", type="primary", use_container_width=True):
    if not name: st.error("성함을 입력해 주세요.")
    else: st.session_state.step = 2; st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: 정신건강 설문 ---
elif st.session_state.step == 2:
 SURVEY_DATA = {
  1: {"title": "📋 PHQ-9 (우울 설문)", "questions": ["1-1. 일을 하는 것에 대한 흥미나 재미가 거의 없음", "1-2. 기분이 가라앉거나 우울하거나 희망이 없다고 느꼈다", "1-3. 잠들기 어렵거나 자주 깨거나 너무 많이 잤다", "1-4. 피곤하고 기력이 거의 없었다", "1-5. 식욕이 저하되거나 과식을 했다", "1-6. 자신이 실패자라고 느끼거나 자신 또는 가족을 실망시켰다", "1-7. 신문을 읽거나 TV를 보는 것과 같은 일에 집중하기 어려웠다", "1-8. 다른 사람들이 알아챌 정도로 너무 느리게 움직이거나 말을 했다", "1-9. 자신을 해치거나 차라리 죽는 것이 낫겠다는 생각을 했다"], "options": ["전혀 아니다", "여러 날 동안", "일주일 이상", "거의 매일"], "key": "PHQ9"},
  2: {"title": "😰 GAD-7 (불안 설문)", "questions": ["2-1. 초조하거나 불안하거나 조마조마하게 느낀다", "2-2. 걱정하는 것을 멈추거나 조절할 수 없다", "2-3. 여러 가지 것들에 대해 걱정을 너무 많이 한다", "2-4. 편하게 있기가 어렵다", "2-5. 너무 안절부절못해서 가만히 있기 힘들다", "2-6. 쉽게 짜증이 나거나 쉽게 성을 낸다", "2-7. 마치 끔찍한 일이 일어날 것처럼 두렵게 느낀다"], "options": ["전혀 아니다", "며칠 동안", "7일 이상", "거의 매일"], "key": "GAD7"},
  3: {"title": "😓 BP1 (스트레스 인지)", "questions": ["3. 평소 일상생활 중 스트레스를 어느 정도 느끼십니까?"], "options": ["1. 거의 느끼지 않음", "2. 조금 느끼는 편이다", "3. 많이 느끼는 편이다", "4. 대단히 많이 느낀다"], "key": "BP1"},
  4: {"title": "💪 EQ5D (삶의 질)", "questions": ["4-1. 운동능력", "4-2. 자기관리", "4-3. 일상활동", "4-4. 통증/불편", "4-5. 불안/우울"], "options_per_question": [["1. 걷는데 지장이 없음", "2. 걷는데 다소 지장이 있음", "3. 종일 누워 있어야 함"], ["1. 목욕이나 옷 입는데 지장 없음", "2. 목욕이나 옷 입는데 다소 지장 있음", "3. 혼자 목욕하거나 옷 입기 힘듦"], ["1. 일상 활동에 지장 없음", "2. 일상 활동에 다소 지장 있음", "3. 일상 활동을 할 수 없음"], ["1. 통증이나 불편감 없음", "2. 다소 통증이나 불편감 있음", "3. 매우 심한 통증이나 불편감 있음"], ["1. 불안하거나 우울하지 않음", "2. 다소 불안하거나 우울함", "3. 매우 불안하거나 우울함"]], "key": "EQ5D"}
 }
 curr = SURVEY_DATA[st.session_state.sub_step]
 q_idx = st.session_state.q_idx
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  st.markdown(f'<h3 style="color:#22c55e;">{curr["title"]}</h3>', unsafe_allow_html=True)
  st.progress((q_idx + 1) / len(curr['questions']))
  st.markdown(f"#### {curr['questions'][q_idx]}")
  opts = curr["options_per_question"][q_idx] if "options_per_question" in curr else curr["options"]
  ans = st.radio("S", opts, key=f"q_{st.session_state.sub_step}_{q_idx}", label_visibility="collapsed")
  st.session_state.survey_answers[curr["key"]][f"q{q_idx}"] = ans
  b1, b2 = st.columns(2)
  with b1:
   if st.button("⬅ 이전 질문", use_container_width=True):
    if q_idx > 0: st.session_state.q_idx -= 1
    elif st.session_state.sub_step > 1: st.session_state.sub_step -= 1; st.session_state.q_idx = len(SURVEY_DATA[st.session_state.sub_step]["questions"]) - 1
    else: st.session_state.step = 1
    st.rerun()
  with b2:
   btn_txt = "다음 질문 ➡" if q_idx < len(curr["questions"]) - 1 else ("다음 설문 ➡" if st.session_state.sub_step < 4 else "분석 결과 보기 🎯")
   if st.button(btn_txt, type="primary", use_container_width=True):
    if q_idx < len(curr["questions"]) - 1: st.session_state.q_idx += 1
    elif st.session_state.sub_step < 4: st.session_state.sub_step += 1; st.session_state.q_idx = 0
    else: st.session_state.step = 3
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: AI 분석 리포트 ---
elif st.session_state.step == 3:
 st.markdown("<h2 style='text-align:center; margin-bottom:30px;'>📊 개인 건강 분석 리포트</h2>", unsafe_allow_html=True)
 u = st.session_state.user_data
 bmi = u['weight'] / ((u['height']/100)**2)
 phq, gad, bp1_score, eq5d = calculate_scores()

 # --- [수정 부분] 정신건강 텍스트 변환 로직 ---
 # 1. PHQ-9 (우울)
 if phq <= 4: phq_text = "정상"
 elif phq <= 9: phq_text = "가벼운 우울증"
 elif phq <= 19: phq_text = "중간 정도의 우울증"
 else: phq_text = "심한 우울증"

 # 2. GAD-7 (불안)
 if gad <= 4: gad_text = "정상"
 elif gad <= 9: gad_text = "가벼운 불안"
 elif gad <= 14: gad_text = "중간 정도의 불안"
 else: gad_text = "심한 불안"

 # 3. 스트레스
 stress_map = {1: "낮음", 2: "보통", 3: "높음", 4: "매우 높음"}
 stress_text = stress_map.get(bp1_score, "보통")

 # 4. 삶의 질 (EQ-5D)
 if eq5d == 1: eq_text = "매우 높음"
 elif eq5d >= 0.899: eq_text = "높음"
 elif eq5d >= 0.8: eq_text = "보통"
 elif eq5d >= 0.7: eq_text = "낮음"
 else: eq_text = "매우 낮음"

 st.markdown(f"""
 <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
  <h3 style="margin:0; color: white;">👤 {u['name']}님의 건강 프로필</h3>
  <p style="margin:5px 0; color: white;">나이: {u['age']}세 | 성별: {u['gender']} | BMI: {bmi:.1f}</p>
  <p style="margin:5px 0; color: white;">우울: <b>{phq_text}</b>({phq}점) | 불안: <b>{gad_text}</b>({gad}점)</p>
  <p style="margin:5px 0; color: white;">스트레스: <b>{stress_text}</b> | 삶의 질: <b>{eq_text}</b>({eq5d:.3f}점)</p>
 </div>
 """, unsafe_allow_html=True)
 
 preds = get_predictions()
 high_risks, mid_risks = [], []
 risk_summary_text = []
 
 for d_name, res in preds.items():
  prob, threshold = res['prob'], res['threshold']
  score = int(prob * 100)
  
  if prob >= threshold: level = "높음"; high_risks.append(d_name)
  elif prob >= threshold * 0.7: level = "중간"; mid_risks.append(d_name)
  else: level = "낮음"
  
  if level in ["높음", "중간"]: risk_summary_text.append(f"{d_name}({level})")
  theme = LEVEL_THEMES[level]
  
  st.markdown(f"""
  <div class="disease-item-card">
   <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <span style="font-weight: bold; font-size: 1.2rem; color: #334155;">{theme['emoji']} {d_name}</span>
    <div style="text-align: right;">
     <span style="color: {theme['color']}; font-weight: bold; font-size: 1.2rem;">{level}</span>
     <span style="color: #64748b; font-size: 0.9rem; margin-left: 8px;">위험도 {score}점</span>
    </div>
   </div>
   <div style="width: 100%; background-color: #f1f5f9; border-radius: 10px; height: 14px; overflow: hidden;">
    <div style="width: {score}%; background-color: {theme['color']}; height: 100%; border-radius: 10px;"></div>
   </div>
   <div style="margin-bottom: 15px;">
 <p style="margin-top: 10px; color: #64748b; font-size: 0.9rem;">
  발병 확률: {prob:.1%} | 기준 임계값: {threshold:.1%}
 </p>
 <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px; line-height: 1.4;">
  * 높음: 임계값 이상 | 중간: 임계값의 70% 이상 | 낮음: 70% 미만
 </p>
</div>
  """, unsafe_allow_html=True)
 
 st.session_state.risks_summary = ", ".join(risk_summary_text) if risk_summary_text else "정상"
 st.write("---")
 st.markdown("### 💡 종합 의견")
 if high_risks: st.error(f"**고위험 질환**: {', '.join(high_risks)} - 전문의 상담을 권장합니다.")
 if mid_risks: st.warning(f"**중위험 질환**: {', '.join(mid_risks)} - 생활습관 개선이 필요합니다.")
 if not high_risks and not mid_risks: st.success("모든 질환이 저위험입니다. 현재 상태를 유지하세요!")
 
 st.write("---")
 c1, c2 = st.columns(2)
 with c1:
  if st.button("🎙️ AI 상담 시작하기", type="primary", use_container_width=True):
   st.session_state.chat_history = [{"role": "ai", "content": f"안녕하세요 {st.session_state.user_data['name']}님. 분석 결과 {st.session_state.risks_summary} 위험이 확인되었습니다. 어떤 점을 도와드릴까요?"}]
   st.session_state.step = 4; st.rerun()
 with c2:
  if st.button("🔄 처음으로 돌아가기", use_container_width=True):
   for key in [k for k in st.session_state.keys() if k != 'db']: del st.session_state[key]
   st.rerun()

# --- STEP 4: AI 음성 챗봇 상담 ---
elif st.session_state.step == 4:
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  head1, head2 = st.columns([3, 1])
  with head1:
   st.subheader("🤖 AI 건강 비서")
  with head2:
   st.session_state.tts_enabled = st.toggle("🔊 음성 답변", value=st.session_state.tts_enabled)
  
  chat_box = st.container()
  for msg in st.session_state.chat_history:
   if msg["role"] == "user":
    st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
   else:
    st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    if "audio" in msg and msg["audio"]:
     st.audio(msg["audio"], format="audio/mp3")
  
  col1, col2 = st.columns([4, 1])
  with col2:
   st.write("🎙️ 음성")
   voice_msg = speech_to_text(language='ko', just_once=True, key='stt_final')
  with col1:
   user_msg = st.chat_input("증상이나 궁금한 점을 물어보세요.")
  
  final_input = voice_msg if voice_msg else user_msg
  if final_input:
   st.session_state.chat_history.append({"role": "user", "content": final_input})
   try:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)
    u = st.session_state.user_data
    phq, gad, _, _ = calculate_scores()
    
    sys_msg = (
        f"당신은 전문 건강 상담사입니다.\n"
        f"상담 대상: {u['name']}님({u['age']}세, {u['gender']}).\n"
        f"가족력 정보: {', '.join(u['family_history']) if u['family_history'] else '없음'}.\n"
        f"현재 분석된 위험 질환: {st.session_state.risks_summary}.\n"
        f"정신건강: 우울 {phq}점, 불안 {gad}점.\n\n"
        f"중요 지침:\n"
        f"1. 사용자가 특정 질환의 위험도가 왜 높은지 물으면 분석 결과 가중치를 기반으로 논리적으로 설명하세요.\n"
        f"2. 모든 답변은 친절하고 전문적인 의학적 근거를 바탕으로 하되, 마지막에는 반드시 예방을 위한 생활 습관 조언을 덧붙이세요.")
    
    with st.chat_message("assistant", avatar="🤖"):
     placeholder = st.empty()
     full_response = ""
     for chunk in llm.stream([SystemMessage(content=sys_msg), HumanMessage(content=final_input)]):
      full_response += chunk.content
      placeholder.markdown(full_response + "▌")
     placeholder.markdown(full_response)
    
    audio_data = None
    if st.session_state.tts_enabled:
     with st.spinner("음성 생성 중..."):
      tts = gTTS(text=full_response, lang='ko')
      audio_fp = io.BytesIO()
      tts.write_to_fp(audio_fp)
      audio_data = audio_fp.getvalue()
    
    st.session_state.chat_history.append({"role": "ai", "content": full_response, "audio": audio_data})
    st.rerun()
   except Exception as e:
    st.error(f"상담 중 오류가 발생했습니다: {e}")
  
  st.write("---")
  foot_col1, foot_col2 = st.columns(2)
  with foot_col1:
   if st.button("⬅ 결과 리포트로 돌아가기", use_container_width=True): 
    st.session_state.step = 3
    st.rerun()
  with foot_col2:
   if st.button("🔄 처음으로 (메인화면)", use_container_width=True):
    for key in [k for k in st.session_state.keys() if k != 'db']: del st.session_state[key]
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)



