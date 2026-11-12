import io
import time

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ----------------------------- Configuration -----------------------------
MODEL_NAME = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash"]  # used only on 503
MAX_CONTEXT_CHARS = 15000  # requested limit; raise (e.g. 400000) for long decks
HISTORY_TURNS = 6          # recent chat messages sent back to the model

st.set_page_config(page_title="Chat & Study Hub", page_icon="📚", layout="wide")


# ----------------------------- Gemini helpers -----------------------------
@st.cache_resource
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def is_503(err: Exception) -> bool:
    code = getattr(err, "code", None)
    text = str(err)
    return code == 503 or "503" in text or "UNAVAILABLE" in text


def call_gemini_safe(client, prompt: str, system_instruction: str = None,
                     max_retries: int = 2) -> str:
    """Try the main model with quick retries; on repeated 503, fall back to other models."""
    config = (types.GenerateContentConfig(system_instruction=system_instruction)
              if system_instruction else None)
    last_err = None
    for model in [MODEL_NAME] + FALLBACK_MODELS:
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                if model != MODEL_NAME:
                    st.toast(f"Main model busy, answered with {model}", icon="ℹ️")
                return response.text or "The model returned an empty response."
            except Exception as e:  # noqa: BLE001
                last_err = e
                if is_503(e):
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    break  # give up on this model, try the next one
                return f"⚠️ Error while calling Gemini: {e}"
    return ("⚠️ All Gemini models are under high demand right now (503). "
            "Please try again in a minute.")


# ----------------------------- PDF helpers -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes: bytes, filename: str):
    """Return (text with [Source | Page] tags, number of pages)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if text:
            parts.append(f"[Source: {filename} | Page {i}]\n{text}")
    return "\n\n".join(parts), len(reader.pages)


def build_context(files):
    texts, total_pages = [], 0
    for f in files:
        text, pages = extract_pdf(f.getvalue(), f.name)
        texts.append(text)
        total_pages += pages
    return "\n\n".join(texts), total_pages


# ----------------------------- Session state -----------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("quiz", "")
st.session_state.setdefault("summary", "")

# ----------------------------- API key -----------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:  # noqa: BLE001
    api_key = None

# ----------------------------- Sidebar -----------------------------
st.sidebar.title("📚 Study Hub")

if api_key:
    st.sidebar.success("✅ GEMINI_API_KEY loaded")
else:
    st.sidebar.error("❌ GEMINI_API_KEY missing in st.secrets")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

st.sidebar.subheader("Study Controls")
num_questions = st.sidebar.slider("Number of Practice Questions", 1, 20, 5)
question_type = st.sidebar.selectbox(
    "Question Type",
    ["Multiple Choice (MCQs)", "True / False", "Open-Ended / Short Answer"],
)
difficulty = st.sidebar.select_slider(
    "Difficulty Level", options=["Easy", "Medium", "Hard"], value="Medium"
)

if st.session_state.messages:
    export_text = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages
    )
    st.sidebar.download_button(
        "⬇️ Export chat history",
        data=export_text,
        file_name="study_chat_history.txt",
        mime="text/plain",
    )
    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------- Main -----------------------------
st.title("📖 Advanced Chat & Study Hub")

if not api_key:
    st.error("Add `GEMINI_API_KEY` to your Streamlit secrets "
             "(`.streamlit/secrets.toml` locally, or App Settings → Secrets on Cloud).")
    st.stop()

if not uploaded_files:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
    st.stop()

client = get_client(api_key)

with st.spinner("Reading PDFs..."):
    full_text, total_pages = build_context(uploaded_files)

if not full_text.strip():
    st.error("No extractable text found. The PDFs may be scanned images.")
    st.stop()

active_context = full_text[:MAX_CONTEXT_CHARS]

c1, c2, c3 = st.columns(3)
c1.metric("Documents processed", len(uploaded_files))
c2.metric("Total characters", f"{len(full_text):,}")
c3.metric("Characters sent to AI", f"{len(active_context):,}")
if len(full_text) > MAX_CONTEXT_CHARS:
    st.warning(
        f"Only the first {MAX_CONTEXT_CHARS:,} characters are used by the AI. "
        "Later pages are not included."
    )

DOC_RULES = (
    "You are a study assistant. Use ONLY the document text provided. "
    "If the answer is not in the documents, say so. "
    "Cite sources as (Source: filename, Page N) using the [Source | Page] tags."
)

tab_chat, tab_quiz, tab_summary = st.tabs(
    ["💬 Chat & Q&A", "📝 Quiz & Practice", "📌 Summary Generator"]
)

# ---- Chat tab ----
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages[-HISTORY_TURNS:-1]
        )
        prompt = (
            f"DOCUMENT TEXT:\n{active_context}\n\n"
            f"RECENT CONVERSATION:\n{history or '(none)'}\n\n"
            f"QUESTION: {question}"
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_gemini_safe(client, prompt, DOC_RULES)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- Quiz tab ----
with tab_quiz:
    st.write(f"**{num_questions}** · **{question_type}** · **{difficulty}**")
    if st.button("Generate Practice Test", type="primary"):
        prompt = (
            f"DOCUMENT TEXT:\n{active_context}\n\n"
            f"Create {num_questions} {difficulty}-level practice questions of type "
            f"'{question_type}' based only on the document. For each question give "
            "the correct answer and a brief explanation. For multiple choice, "
            "provide options A-D. Put all answers in an 'Answer Key' after the "
            "questions, with page citations where possible."
        )
        with st.spinner("Generating quiz..."):
            st.session_state.quiz = call_gemini_safe(client, prompt, DOC_RULES)
    if st.session_state.quiz:
        st.markdown(st.session_state.quiz)

# ---- Summary tab ----
with tab_summary:
    if st.button("Generate Summary", type="primary"):
        prompt = (
            f"DOCUMENT TEXT:\n{active_context}\n\n"
            "Write a well-structured, comprehensive summary: an overview, key "
            "topics with headings, important definitions or facts as bullets, "
            "and a short conclusion. Cite pages where useful."
        )
        with st.spinner("Summarizing..."):
            st.session_state.summary = call_gemini_safe(client, prompt, DOC_RULES)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
