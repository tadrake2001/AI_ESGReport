import tiktoken
import streamlit as st
import requests
import pandas as pd
import time
import tempfile
import fitz
import re
import json

# Bỏ chọn batch size, mặc định là 10
batch_size = 10
MAX_TOKENS_LIMIT = 4096
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

EXCLUDE_PATTERNS = [
    r'(?i)(email|mailto|contact|phone|website|www\.|@)',
    r'\bpage\b\s*\d+',
    r'(?i)(copyright|disclaimer)'
]

HEADER_PATTERNS = [
    r'^\s*\d+(\.\d+)*\s*$',
    r'^\s*(tổng quan|giới thiệu|mục tiêu|phạm vi|cam kết|chính sách).*$',
    r'^\s*[A-Z ]{3,}\s*$'
]

def clean_text(text):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if any(re.search(pat, line) for pat in EXCLUDE_PATTERNS):
            continue
        if len(line.strip()) > 0:
            clean_lines.append(line.strip())
    return " ".join(clean_lines)

def is_header_like(text):
    for pattern in HEADER_PATTERNS:
        if re.match(pattern, text.strip(), flags=re.IGNORECASE):
            return True
    return False

def estimate_tokens(text):
    return len(enc.encode(text))

def batch_sentences(sentences, batch_size=10):
    return [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

def format_input_for_model(batch):
    return {
        "instruction": "Hãy xác định các câu có dấu hiệu phóng đại trong báo cáo ESG và giải thích lý do.",
        "input": batch
    }

def query_batch(batch):
    formatted_input = format_input_for_model(batch)
    headers = {"Content-Type": "application/json"}
    prompt = json.dumps(formatted_input, ensure_ascii=False, indent=2)
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False
    }
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content']
    except Exception as e:
        return [("", {"label": "lỗi", "reason": str(e)})]

    results = []
    pattern = r"^- (.+?)\n\s*→\s*(.+)"
    matches = re.findall(pattern, result, re.MULTILINE)
    for sentence, reason in matches:
        results.append((sentence.strip(), {"label": "phóng đại", "reason": reason.strip()}))
    return results

# Giao diện
st.set_page_config(page_title="Phân tích ESG", layout="wide")
st.title("\U0001F4C4 Phân tích Báo cáo ESG và Tìm câu phóng đại")

uploaded_file = st.file_uploader("\U0001F4C4 Tải báo cáo ESG dạng PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.success("✅ Đã tải lên thành công!")

    with fitz.open(tmp_path) as doc:
        all_text = "\n".join(clean_text(page.get_text()) for page in doc)

    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    filtered_sentences = [s.strip() for s in sentences if 30 < len(s.strip()) < 400 and not is_header_like(s)]
    st.session_state["edited_sentences"] = filtered_sentences
    st.write(f"🔎 Đã phát hiện {len(filtered_sentences)} câu hợp lệ từ báo cáo.")

if "edited_sentences" in st.session_state:
    st.subheader("✍️ Chỉnh sửa các nhóm câu trước khi phân tích")
    batches = batch_sentences(st.session_state["edited_sentences"], batch_size=batch_size)
    updated_batches = []
    batch_included = []

    for i, batch in enumerate(batches):
        include_group = st.checkbox(f"✅ Gửi nhóm {i+1}", value=True, key=f"include_batch_{i}")
        batch_included.append(include_group)

        with st.expander(f"📦 Nhóm {i+1}"):
            updated = []
            for j, sentence in enumerate(batch):
                edited = st.text_area(f"Câu {j+1}:", sentence, key=f"group_{i}_sent_{j}")
                if edited.strip():
                    updated.append(edited.strip())
            updated_batches.append(updated)

    if st.button("\U0001F50D Gửi và lọc các câu phóng đại"):
        st.info("Đang phân tích và lọc các câu phóng đại theo nhóm...")
        progress_bar = st.progress(0)
        exaggerated = []
        start_time = time.time()

        for i, (batch, included) in enumerate(zip(updated_batches, batch_included)):
            if not included:
                continue

            formatted_input = format_input_for_model(batch)
            input_text = json.dumps(formatted_input, ensure_ascii=False, indent=2)
            token_count = estimate_tokens(input_text)
            with st.expander(f"\U0001F4E4 Input nhóm {i+1} ({token_count} tokens)"):
                st.code(input_text, language="json")
                if token_count > MAX_TOKENS_LIMIT:
                    st.warning(f"⚠️ Nhóm này vượt quá giới hạn {MAX_TOKENS_LIMIT} tokens, có thể gây lỗi hoặc bị cắt nội dung.")

            result = query_batch(batch)
            for sentence, res in result:
                if res['label'] == "phóng đại":
                    exaggerated.append({"Câu": sentence, "Lý do": res['reason']})
            progress_bar.progress((i + 1) / len(updated_batches))

        elapsed = time.time() - start_time
        avg_per_batch = elapsed / len(updated_batches) if updated_batches else 0
        st.success(f"✅ Hoàn tất trong {elapsed:.1f} giây (trung bình {avg_per_batch:.1f} giây mỗi nhóm)")

        if exaggerated:
            st.subheader(f"⚠️ Phát hiện {len(exaggerated)} câu có dấu hiệu phóng đại")

            st.markdown("""
                    #### 📌 Danh sách câu có dấu hiệu phóng đại
                    """)
            for item in exaggerated:
                st.markdown(
                    f"""
                            <div style='background-color:#fff9f5;padding:0.75em 1em;margin-bottom:1em;border-left:6px solid #f06a36;border-radius:6px'>
                                <div style='color:#222;font-weight:600;margin-bottom:0.25em'>📝 {item['Câu']}</div>
                                <div style='color:#555'>➡️ <i>{item['Lý do']}</i></div>
                            </div>
                            """,
                    unsafe_allow_html=True
                )
            summary = "\n".join(
                f"- {item['Câu']}\n  → {item['Lý do']}"
                for item in exaggerated
            )
            df = pd.DataFrame(exaggerated)
            st.download_button(
                label="\U0001F4E5 Tải danh sách câu phóng đại",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="esg_phong_dai.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ Không có câu phóng đại nào được phát hiện.")