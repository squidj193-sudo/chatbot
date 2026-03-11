"""
Streamlit Web GUI — Gemini 多模態聊天程式
支援：純文字對話、圖片 (JPG/PNG)、PDF、TXT 檔案上傳
具備對話記憶，可匯出 JSON 對話紀錄
"""

import os
import sys
import json
import base64
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader

# ── 頁面設定（必須是第一個 Streamlit 指令）────────────────
st.set_page_config(
    page_title="Gemini Chat",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自訂 CSS ──────────────────────────────────────────
st.markdown("""
<style>
/* ---------- 全域排版 ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- 主區域 ---------- */
.main .block-container {
    max-width: 860px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------- 側邊欄 ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}

/* ---------- 聊天氣泡 ---------- */
.stChatMessage {
    border-radius: 16px !important;
    padding: 0.8rem 1.2rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ---------- 按鈕 ---------- */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}

/* ---------- 檔案上傳區 ---------- */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}
[data-testid="stFileUploader"] > div {
    border-radius: 12px;
}

/* ---------- 下載按鈕 ---------- */
.stDownloadButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
}

/* ---------- 狀態標籤 ---------- */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 4px;
}
.badge-image { background: #312e81; color: #a5b4fc; }
.badge-pdf   { background: #7f1d1d; color: #fca5a5; }
.badge-txt   { background: #064e3b; color: #6ee7b7; }

/* ---------- Sidebar 統計卡片 ---------- */
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── 初始化 ─────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("找不到 **GEMINI_API_KEY**，請在 `.env` 檔案中設定。")
    st.stop()


@st.cache_resource
def get_llm():
    """建立並快取 LLM 實例。"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
    )


llm = get_llm()

# ── Session State 初始化 ──────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="你是一位友善且專業的 AI 助手。"
                              "當使用者提供檔案內容（圖片、PDF、TXT）時，"
                              "請仔細分析並回答使用者的問題。"
                              "請使用繁體中文回答。")
    ]
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "pending_file" not in st.session_state:
    st.session_state.pending_file = None
if "file_counter" not in st.session_state:
    st.session_state.file_counter = 0


# ── 工具函式 ─────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


def append_log(role: str, content: str, file_name: str | None = None,
               file_type: str | None = None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "content": sanitize_text(content),
    }
    if file_name:
        entry["file"] = file_name
        entry["file_type"] = file_type
    st.session_state.chat_log.append(entry)


def get_json_download() -> str:
    return json.dumps(st.session_state.chat_log, ensure_ascii=False, indent=2)


def process_image(uploaded_file) -> dict:
    data = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    ext = Path(uploaded_file.name).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(ext, "image/jpeg")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{data}"},
    }


def process_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        full_text = "\n\n".join(
            f"[第 {i+1} 頁]\n{page.page_content}" for i, page in enumerate(pages)
        )
        return full_text
    finally:
        os.unlink(tmp_path)


def process_txt(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8")


def detect_file_type(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png"}:
        return "image"
    elif ext == ".pdf":
        return "pdf"
    elif ext == ".txt":
        return "txt"
    return None


def file_badge(file_type: str, file_name: str) -> str:
    badge_class = {"image": "badge-image", "pdf": "badge-pdf", "txt": "badge-txt"}
    icons = {"image": "🖼️", "pdf": "📄", "txt": "📝"}
    cls = badge_class.get(file_type, "")
    icon = icons.get(file_type, "📎")
    return f'<span class="status-badge {cls}">{icon} {file_name}</span>'


# ── 側邊欄 ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✨ Gemini Chat")
    st.caption("多模態 AI 聊天助手")

    st.markdown("---")

    # 統計資訊
    user_count = sum(1 for m in st.session_state.display_messages if m["role"] == "user")
    ai_count = sum(1 for m in st.session_state.display_messages if m["role"] == "ai")
    file_count = sum(1 for m in st.session_state.display_messages if m.get("file_type"))

    cols = st.columns(3)
    with cols[0]:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{user_count}</div>'
                    f'<div class="stat-label">你的訊息</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{ai_count}</div>'
                    f'<div class="stat-label">AI 回覆</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{file_count}</div>'
                    f'<div class="stat-label">檔案</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 檔案上傳
    st.markdown("### 📎 上傳檔案")
    uploaded_file = st.file_uploader(
        "支援 JPG / PNG / PDF / TXT",
        type=["jpg", "jpeg", "png", "pdf", "txt"],
        key=f"file_uploader_{st.session_state.file_counter}",
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_type = detect_file_type(uploaded_file.name)
        if file_type:
            st.session_state.pending_file = {
                "file": uploaded_file,
                "name": uploaded_file.name,
                "type": file_type,
            }
            st.success(f"已載入: **{uploaded_file.name}**")
            if file_type == "image":
                st.image(uploaded_file, use_container_width=True)
            st.info("💡 在下方輸入訊息，即可針對檔案提問")

    st.markdown("---")

    # 匯出對話紀錄
    st.markdown("### 💾 對話紀錄")
    if st.session_state.chat_log:
        filename = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
        st.download_button(
            label="📥 下載 JSON",
            data=get_json_download(),
            file_name=filename,
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("尚無對話紀錄")

    st.markdown("---")

    # 清除對話
    if st.button("🗑️ 清除對話", use_container_width=True):
        st.session_state.chat_history = [
            SystemMessage(content="你是一位友善且專業的 AI 助手。"
                                  "當使用者提供檔案內容（圖片、PDF、TXT）時，"
                                  "請仔細分析並回答使用者的問題。"
                                  "請使用繁體中文回答。")
        ]
        st.session_state.chat_log = []
        st.session_state.display_messages = []
        st.session_state.pending_file = None
        st.session_state.file_counter += 1
        st.rerun()


# ── 主聊天區 ─────────────────────────────────────────────
st.markdown("# 💬 Gemini 多模態聊天")
st.caption("支援純文字對話，也可透過側邊欄上傳圖片 / PDF / TXT 檔案")

# 顯示歷史訊息
for msg in st.session_state.display_messages:
    role_icon = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role_icon):
        # 如果有附檔標記
        if msg.get("file_type") and msg.get("file_name"):
            st.markdown(file_badge(msg["file_type"], msg["file_name"]),
                       unsafe_allow_html=True)
        # 如果是圖片，顯示預覽
        if msg.get("image_data"):
            st.image(msg["image_data"], width=300)
        st.markdown(msg["content"])

# 聊天輸入
if prompt := st.chat_input("輸入訊息（若已上傳檔案，可直接針對檔案提問）..."):
    pending = st.session_state.pending_file
    file_type = pending["type"] if pending else None
    file_name = pending["name"] if pending else None

    # ── 使用者訊息 ──
    user_display = {"role": "user", "content": prompt}

    if pending:
        user_display["file_type"] = file_type
        user_display["file_name"] = file_name

    # 準備 LangChain 訊息
    if pending and file_type == "image":
        image_data = process_image(pending["file"])
        message_content = [
            {"type": "text", "text": prompt},
            image_data,
        ]
        human_msg = HumanMessage(content=message_content)
        user_display["image_data"] = pending["file"].getvalue()
        append_log("user", prompt, file_name, file_type)

    elif pending and file_type == "pdf":
        pdf_text = process_pdf(pending["file"])
        full_prompt = f"以下是一份 PDF 文件的內容：\n\n{pdf_text}\n\n{prompt}"
        human_msg = HumanMessage(content=full_prompt)
        append_log("user", full_prompt, file_name, file_type)

    elif pending and file_type == "txt":
        txt_content = process_txt(pending["file"])
        full_prompt = f"以下是一份文字檔的內容：\n\n{txt_content}\n\n{prompt}"
        human_msg = HumanMessage(content=full_prompt)
        append_log("user", full_prompt, file_name, file_type)

    else:
        human_msg = HumanMessage(content=prompt)
        append_log("user", prompt)

    st.session_state.display_messages.append(user_display)
    st.session_state.chat_history.append(human_msg)

    # 顯示使用者訊息
    with st.chat_message("user"):
        if user_display.get("file_type") and user_display.get("file_name"):
            st.markdown(file_badge(user_display["file_type"], user_display["file_name"]),
                       unsafe_allow_html=True)
        if user_display.get("image_data"):
            st.image(user_display["image_data"], width=300)
        st.markdown(prompt)

    # 清除 pending file
    if pending:
        st.session_state.pending_file = None
        st.session_state.file_counter += 1

    # ── AI 回覆 ──
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = llm.invoke(st.session_state.chat_history)
                ai_text = response.content
            except Exception as e:
                ai_text = f"⚠️ 無法取得回覆：{e}"

        st.markdown(ai_text)

    ai_msg = AIMessage(content=ai_text)
    st.session_state.chat_history.append(ai_msg)
    append_log("ai", ai_text)
    st.session_state.display_messages.append({
        "role": "ai",
        "content": ai_text,
    })

    # 自動儲存 JSON 到工作目錄
    auto_save_path = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
    with open(auto_save_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_log, f, ensure_ascii=False, indent=2)
