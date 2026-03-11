"""
多模態聊天程式 — 使用 LangChain + Gemini 2.5 Flash
支援：純文字對話、圖片 (JPG/PNG)、PDF、TXT 檔案輸入
對話紀錄自動儲存為 JSON
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader

# ── 修正 Windows 終端編碼 ──────────────────────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# ── 初始化 ─────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[ERROR] 找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
    sys.exit(1)

# LangChain Gemini 模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
)

# 對話歷史（LangChain Messages）
chat_history: list = []
# 用於 JSON 持久化的紀錄
chat_log: list[dict] = []

# 支援的檔案副檔名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt"}


# ── 工具函式 ─────────────────────────────────────────────
def sanitize_text(text: str) -> str:
    """移除無法被 JSON 序列化的 surrogate 字元。"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def detect_file(user_input: str) -> tuple[str | None, str | None]:
    """
    判斷使用者輸入是否為檔案路徑。
    回傳 (file_path, file_type) 或 (None, None)。
    """
    text = user_input.strip().strip('"').strip("'")
    path = Path(text)
    if path.is_file():
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return str(path), "image"
        elif ext in PDF_EXTENSIONS:
            return str(path), "pdf"
        elif ext in TEXT_EXTENSIONS:
            return str(path), "txt"
    return None, None


def load_image_as_base64(file_path: str) -> dict:
    """將圖片讀取並轉為 base64，回傳 LangChain 可用的 image 內容格式。"""
    ext = Path(file_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{data}"},
    }


def load_pdf_text(file_path: str) -> str:
    """使用 PyPDFLoader 載入 PDF 並回傳所有頁面的文字。"""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n\n".join(
        f"[第 {i+1} 頁]\n{page.page_content}" for i, page in enumerate(pages)
    )
    return full_text


def load_txt(file_path: str) -> str:
    """讀取純文字檔內容。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def append_log(role: str, content: str, file_path: str | None = None,
               file_type: str | None = None):
    """將一筆對話紀錄加入 chat_log。"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "content": sanitize_text(content),
    }
    if file_path:
        entry["file"] = file_path
        entry["file_type"] = file_type
    chat_log.append(entry)


def save_chat_log():
    """將 chat_log 儲存為 JSON，檔名格式: chat_YYYYMMDD_HHMMSS.json"""
    if not chat_log:
        return
    filename = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chat_log, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] 對話紀錄已儲存至: {filename}")


# ── 主流程 ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("Gemini 多模態聊天程式（具備對話記憶）")
    print("  支援：純文字 / 圖片 (JPG/PNG) / PDF / TXT")
    print("  輸入檔案路徑即可分析檔案內容")
    print("  輸入 'exit' 結束對話並儲存紀錄")
    print("=" * 55)
    print()

    system_msg = SystemMessage(
        content="你是一位友善且專業的 AI 助手。"
                "當使用者提供檔案內容（圖片、PDF、TXT）時，"
                "請仔細分析並回答使用者的問題。"
                "請使用繁體中文回答。"
    )
    chat_history.append(system_msg)

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # 離開指令
            if user_input.lower() in ("exit", "quit"):
                print("\n再見！")
                break

            # 偵測是否為檔案路徑
            file_path, file_type = detect_file(user_input)

            if file_path:
                # ── 檔案處理 ──
                print(f"[FILE] 偵測到檔案: {file_path} ({file_type})")

                if file_type == "image":
                    question = input("  請問你想問關於這張圖片什麼？（直接 Enter 則請 AI 描述圖片）: ").strip()
                    if not question:
                        question = "請詳細描述這張圖片的內容。"

                    image_data = load_image_as_base64(file_path)
                    message_content = [
                        {"type": "text", "text": question},
                        image_data,
                    ]
                    human_msg = HumanMessage(content=message_content)
                    append_log("user", question, file_path, file_type)

                elif file_type == "pdf":
                    pdf_text = load_pdf_text(file_path)
                    page_count = pdf_text.count("[第")
                    print(f"  已載入 PDF（共 {page_count} 頁）")
                    question = input("  請問你想問關於這份 PDF 什麼？（直接 Enter 則請 AI 摘要）: ").strip()
                    if not question:
                        question = "請摘要這份文件的重點內容。"

                    prompt = f"以下是一份 PDF 文件的內容：\n\n{pdf_text}\n\n{question}"
                    human_msg = HumanMessage(content=prompt)
                    append_log("user", prompt, file_path, file_type)

                elif file_type == "txt":
                    txt_content = load_txt(file_path)
                    print(f"  已載入文字檔（共 {len(txt_content)} 字元）")
                    question = input("  請問你想問關於這份檔案什麼？（直接 Enter 則請 AI 摘要）: ").strip()
                    if not question:
                        question = "請摘要這份文件的重點內容。"

                    prompt = f"以下是一份文字檔的內容：\n\n{txt_content}\n\n{question}"
                    human_msg = HumanMessage(content=prompt)
                    append_log("user", prompt, file_path, file_type)
                else:
                    print("[WARN] 不支援的檔案類型。")
                    continue
            else:
                # ── 純文字對話 ──
                human_msg = HumanMessage(content=user_input)
                append_log("user", user_input)

            # 加入歷史並送出
            chat_history.append(human_msg)

            print("\nAI: ", end="", flush=True)
            try:
                response = llm.invoke(chat_history)
                ai_text = response.content
                print(ai_text)
            except Exception as e:
                ai_text = f"[ERROR] 無法取得回覆: {e}"
                print(ai_text)

            # 儲存 AI 回覆
            ai_msg = AIMessage(content=ai_text)
            chat_history.append(ai_msg)
            append_log("ai", ai_text)
            print()

    except KeyboardInterrupt:
        print("\n\n[WARN] 偵測到中斷信號。")

    finally:
        save_chat_log()


if __name__ == "__main__":
    main()
