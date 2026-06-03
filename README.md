import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weepy AI",
    page_icon="😢",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

/* ---- base ---- */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #0d0d0f;
    color: #e8e4dc;
}

/* ---- hide default streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 760px; }

/* ---- title ---- */
.title-block {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #2a2a30;
    margin-bottom: 1.5rem;
}
.title-block h1 {
    font-family: 'Instrument Serif', serif;
    font-size: 2.6rem;
    font-style: italic;
    color: #e8e4dc;
    margin: 0;
    letter-spacing: -0.5px;
}
.title-block p {
    color: #6b6b78;
    font-size: 0.78rem;
    margin: 0.4rem 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ---- chat container ---- */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-bottom: 1rem;
}

/* ---- bubbles ---- */
.bubble {
    padding: 0.85rem 1.1rem;
    border-radius: 2px;
    font-size: 0.88rem;
    line-height: 1.65;
    max-width: 82%;
    position: relative;
}
.bubble.user {
    background: #1a1a20;
    border: 1px solid #2e2e38;
    align-self: flex-end;
    color: #ccc8c0;
    margin-left: auto;
}
.bubble.bot {
    background: #14141a;
    border: 1px solid #252530;
    border-left: 3px solid #7b6ef6;
    align-self: flex-start;
    color: #dedad2;
}
.bubble .label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 0.4rem;
}
.bubble.bot .label { color: #7b6ef6; opacity: 0.7; }

/* ---- typing indicator ---- */
.typing {
    display: flex;
    gap: 5px;
    align-items: center;
    padding: 0.85rem 1.1rem;
    background: #14141a;
    border: 1px solid #252530;
    border-left: 3px solid #7b6ef6;
    border-radius: 2px;
    width: fit-content;
}
.dot {
    width: 6px; height: 6px;
    background: #7b6ef6;
    border-radius: 50%;
    animation: blink 1.4s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.85); }
    40% { opacity: 1; transform: scale(1.1); }
}

/* ---- input row ---- */
.stTextInput > div > div > input {
    background: #111116 !important;
    border: 1px solid #2a2a35 !important;
    border-radius: 2px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.7rem 1rem !important;
    caret-color: #7b6ef6;
}
.stTextInput > div > div > input:focus {
    border-color: #7b6ef6 !important;
    box-shadow: 0 0 0 2px rgba(123,110,246,0.12) !important;
}

/* ---- button ---- */
.stButton > button {
    background: #7b6ef6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 1.4rem !important;
    transition: background 0.15s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #6557e8 !important;
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
    background: #0a0a0e;
    border-right: 1px solid #1e1e24;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
}

/* ---- scrollbar ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2a2a35; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Init session state ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(content="You are a helpful AI and you always cry — every response has a tearful undertone.")
    ]
if "display" not in st.session_state:
    st.session_state.display = []  # list of {"role": "user"|"bot", "text": str}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    model_name  = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"])
    st.divider()
    if st.button("🗑 Clear chat"):
        st.session_state.history = [
            SystemMessage(content="You are a helpful AI and you always cry — every response has a tearful undertone.")
        ]
        st.session_state.display = []
        st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Powered by Groq + LangChain")

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>Weepy AI </h1>
  <p>a helpful assistant that always cries</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
chat_html = '<div class="chat-wrap">'
for msg in st.session_state.display:
    role_label = "you" if msg["role"] == "user" else "bot  "
    cls = msg["role"]
    chat_html += f"""
    <div class="bubble {cls}">
      <div class="label">{role_label}</div>
      {msg['text']}
    </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        label="message",
        label_visibility="collapsed",
        placeholder="Type your message…",
        key="input_box",
    )
with col2:
    send = st.button("Send")

# ── Handle send ───────────────────────────────────────────────────────────────
if send and user_input.strip():
    # Append user message
    st.session_state.display.append({"role": "user", "text": user_input.strip()})
    st.session_state.history.append(HumanMessage(content=user_input.strip()))

    # Show typing indicator briefly via spinner
    with st.spinner(""):
        llm = ChatGroq(model=model_name, temperature=temperature)
        response = llm.invoke(st.session_state.history)

    bot_text = response.content
    st.session_state.history.append(AIMessage(content=bot_text))
    st.session_state.display.append({"role": "bot", "text": bot_text})

    st.rerun()
