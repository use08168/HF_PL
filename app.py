# app.py
# -*- coding: utf-8 -*-
"""
🧠 기억 번역 순서 게임 — 완전 오프라인 버전 (Streamlit)
- 로컬 모델만 사용 (Hugging Face에 접속하지 않음)
- 생성: ./models/flan-t5-small (google/flan-t5-small 내려받아 배치)
- 번역: ./models/m2m100_418M  (facebook/m2m100_418M 내려받아 배치)
- 기능: 랜덤 단어 → 짧은 영어 문장 생성 → 한국어 번역(힌트) → 단어 셔플 → 순서 맞히기
"""

import os, re, random, time
from typing import List

import streamlit as st
import pandas as pd

# 🔒 외부 접속 완전 차단 (오프라인 보증)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ====== 경로/설정 ======
FLAN_DIR = "./models/flan-t5-small"
M2M_DIR  = "./models/m2m100_418M"

CSV_PATH = "Concreteness_english.csv"
WORD_COL = "Word"

MAX_WORDS       = 6     # 생성 문장 최대 단어 수(난이도)
MAX_NEW_TOKENS  = 24
RETRIES_GEN     = 6     # 단어 포함/길이 조건 재시도
RETRIES_WORDS   = 50    # 적당한 길이 문장 뽑기 재시도

WORD_RE = re.compile(r"[A-Za-z']+")

# ====== Streamlit 기본 ======
st.set_page_config(page_title="기억 번역 순서 게임 (오프라인)", page_icon="🧠", layout="centered")
st.title("🧠 기억 번역 순서 게임 (오프라인)")
st.caption("모델과 데이터는 로컬에서만 읽습니다. 외부 네트워크 접속 없음.")

# ====== 경로 존재 점검 ======
missing = [p for p in [FLAN_DIR, M2M_DIR] if not os.path.isdir(p)]
if missing:
    st.error("다음 로컬 모델 폴더가 없습니다. 모델 폴더를 준비해 주세요:")
    for p in missing:
        st.code(os.path.abspath(p))
    st.stop()

# ====== 유틸 ======
def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)

def enforce_sentence_end(s: str) -> str:
    s = s.strip()
    if not re.search(r"[.!?]$", s):
        s += "."
    return s

def within_limit(tokens: List[str], lo=3, hi=MAX_WORDS) -> bool:
    return lo <= len(tokens) <= hi

def is_tautology(sent: str, word: str) -> bool:
    """'river is a river' 같은 정의문/중복 패턴 회피"""
    w = re.escape(word.lower())
    return bool(re.search(rf"\b{w}\b\s+is\s+(a|the)?\s*\b{w}\b", sent.lower()))

# ====== 모델 로딩 (로컬에서만) ======
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, set_seed
set_seed(1337)

@st.cache_resource(show_spinner=False)
def load_generator_local():
    tok = AutoTokenizer.from_pretrained(FLAN_DIR, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(FLAN_DIR, local_files_only=True)
    return pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)

@st.cache_resource(show_spinner=False)
def load_en2ko_local():
    tok = AutoTokenizer.from_pretrained(M2M_DIR, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(M2M_DIR, local_files_only=True)
    # M2M100은 src_lang/tgt_lang 필수
    return pipeline("translation", model=mdl, tokenizer=tok,
                    src_lang="en", tgt_lang="ko", device=-1)

# ====== 데이터 로딩 ======
@st.cache_data(show_spinner=False)
def load_wordlist() -> List[str]:
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if WORD_COL in df.columns:
                words = (df[WORD_COL].astype(str).str.strip().str.lower()
                         .dropna().unique().tolist())
                words = [w for w in words if w.isalpha() and len(w) >= 2]
            else:
                words = []
        except Exception:
            words = []
    else:
        words = []
    if not words:
        # 폴백 샘플
        words = ["apple", "river", "music", "future", "pattern", "me", "friend", "time"]
    return words

WORDS = load_wordlist()

# ====== 생성/번역 ======
def build_prompt(word: str) -> str:
    # 단어 1회만, 정의문 회피 지시 추가
    return (f"Write one natural English sentence under {MAX_WORDS} words "
            f"that uses the word '{word}' exactly once. "
            f"Avoid definitions like \"{word} is a {word}\" and keep it conversational.")

def generate_sentence_with_word(gen_pipe, word: str) -> str:
    word = word.lower()
    for _ in range(RETRIES_GEN):
        out = gen_pipe(build_prompt(word),
                       max_new_tokens=MAX_NEW_TOKENS,
                       do_sample=False, num_return_sequences=1)[0]["generated_text"]
        out = enforce_sentence_end(out)
        toks = tokenize_words(out)
        has_word = re.search(rf"\b{re.escape(word)}\b", out, flags=re.IGNORECASE) is not None
        if has_word and within_limit(toks) and not is_tautology(out, word):
            # 단어 2번 이상 등장도 걸러보기(지시 위반 케이스)
            if len(re.findall(rf"\b{re.escape(word)}\b", out, flags=re.IGNORECASE)) <= 2:
                return out
    # 폴백(항상 단어 포함, 짧고 자연스럽게)
    if word == "me":
        return "This is me."
    return f"I like {word}."

def translate_en2ko(trans_pipe, text: str) -> str:
    return trans_pipe(text, max_new_tokens=MAX_NEW_TOKENS)[0]["translation_text"]

# ====== 시각적 로딩: 데이터+모델 ======
def load_resources_ui():
    with st.status("리소스 로드 중…", expanded=True) as status:
        t0 = time.perf_counter()
        status.write("① 단어 목록 로드")
        st.session_state.WORDS = WORDS
        st.dataframe(pd.DataFrame({"Word": WORDS[:50]}),
                     use_container_width=True, height=200)
        status.write(f"→ 단어 {len(WORDS):,}개 준비")

        status.write("② 생성 모델 로드 (flan-t5-small, 오프라인)")
        st.session_state.gen = load_generator_local()

        status.write("③ 번역 모델 로드 (m2m100_418M, 오프라인)")
        st.session_state.tr  = load_en2ko_local()

        status.write("④ 워밍업")
        _ = st.session_state.gen("Write a short sentence with 'time'.", max_new_tokens=12)
        _ = st.session_state.tr("Hello world.", max_new_tokens=12)

        status.update(label=f"리소스 로드 완료 ({time.perf_counter()-t0:.1f}s)", state="complete")
    st.session_state.models_ready = True

# ====== 게임 라운드 ======
def new_round():
    WORDS_LOCAL = st.session_state.WORDS
    for _ in range(RETRIES_WORDS):
        w = random.choice(WORDS_LOCAL)
        s = generate_sentence_with_word(st.session_state.gen, w)
        toks = tokenize_words(s)
        if within_limit(toks):
            break
    else:
        w, s, toks = "me", "This is me.", ["This", "is", "me"]

    shuffled = toks[:]
    random.shuffle(shuffled)

    st.session_state.round_id     = st.session_state.get("round_id", 0) + 1
    st.session_state.word         = w
    st.session_state.sent_en      = s
    st.session_state.sent_ko      = translate_en2ko(st.session_state.tr, s)
    st.session_state.tokens       = toks
    st.session_state.shuffled     = shuffled
    st.session_state.selected_idx = []
    st.session_state.correct      = None
    st.session_state.hint_on      = False

    if "score" not in st.session_state:
        st.session_state.score = {"correct": 0, "total": 0}

# ====== 사이드바: 로드 버튼 ======
if "models_ready" not in st.session_state:
    st.session_state.models_ready = False
if "WORDS" not in st.session_state:
    st.session_state.WORDS = None

with st.sidebar:
    st.header("⚙️ 리소스")
    if not st.session_state.models_ready:
        if st.button("리소스 로드", use_container_width=True):
            load_resources_ui()
    else:
        st.success("모델/데이터 준비 완료(오프라인)")
        st.write(f"단어 수: **{len(st.session_state.WORDS):,}**")
        st.write("· 생성: flan-t5-small (local)")
        st.write("· 번역: m2m100_418M (local)")

# 준비 안 됐으면 중단
if not st.session_state.models_ready:
    st.info("사이드바에서 **리소스 로드**를 먼저 눌러주세요.")
    st.stop()

# 첫 라운드
if "round_id" not in st.session_state:
    new_round()

# ====== 본문 UI ======
c1, c2, c3 = st.columns([1, 3, 1])
with c1: st.metric("라운드", st.session_state.round_id)
with c2: st.metric("한국어 힌트", st.session_state.sent_ko)
with c3: st.metric("단어 수", len(st.session_state.tokens))

st.divider()
st.subheader("단어를 순서대로 눌러 문장을 완성하세요")

shuf = st.session_state.shuffled
sel  = st.session_state.selected_idx
cols = st.columns(len(shuf))
for i, tok in enumerate(shuf):
    disabled = (i in sel) or (st.session_state.correct is True)
    with cols[i]:
        if st.button(tok, key=f"tok_{st.session_state.round_id}_{i}",
                     disabled=disabled, use_container_width=True):
            if i not in st.session_state.selected_idx:
                st.session_state.selected_idx.append(i)
            st.rerun()

chosen = [shuf[i] for i in sel]
st.write("당신의 선택:", " ".join(chosen) if chosen else "—")

b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    if st.button("선택 초기화", use_container_width=True):
        st.session_state.selected_idx = []
        st.session_state.correct = None
        st.rerun()
with b2:
    if st.button("재셔플", use_container_width=True):
        random.shuffle(st.session_state.shuffled)
        st.session_state.selected_idx = []
        st.session_state.correct = None
        st.rerun()
with b3:
    if st.button("힌트 토글", use_container_width=True):
        st.session_state.hint_on = not st.session_state.hint_on
        st.rerun()
with b4:
    if st.button("정답 확인", type="primary", use_container_width=True):
        if len(st.session_state.selected_idx) == len(st.session_state.tokens):
            pred = [shuf[i] for i in st.session_state.selected_idx]
            gold = st.session_state.tokens
            ok = ([p.lower() for p in pred] == [g.lower() for g in gold])
            st.session_state.correct = ok
            st.session_state.score["total"] += 1
            if ok: st.session_state.score["correct"] += 1
        else:
            st.warning("모든 단어를 순서대로 선택해 주세요!")
        st.rerun()
with b5:
    if st.button("다음 라운드 ▶", use_container_width=True):
        new_round()
        st.rerun()

st.divider()
if st.session_state.correct is True:
    st.success("✅ 정답!  원문: " + st.session_state.sent_en)
elif st.session_state.correct is False:
    st.error("❌ 오답!  원문: " + st.session_state.sent_en)

if st.session_state.hint_on and st.session_state.correct is not True:
    st.info("힌트(첫 단어): **" + st.session_state.tokens[0] + "**")

st.divider()
sc = st.session_state.score
acc = (sc["correct"] / sc["total"] * 100) if sc["total"] else 0.0
st.write(f"**스코어:** {sc['correct']} / {sc['total']}  (정확도 {acc:.1f}%)")
