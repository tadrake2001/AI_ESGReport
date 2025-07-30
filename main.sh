#!/bin/bash

echo "======================================================"
echo "ESG LLM DETECTOR - FULL PIPELINE"
echo "======================================================"

# LƯU Ý: Hãy chắc chắn bạn đã cài đặt môi trường và các thư viện cần thiết trước khi chạy.
# Xem hướng dẫn chi tiết ở "Bước 0: Thiết Lập Môi Trường".

# Bước 1: Fine-tune mô hình LLM bằng LoRA và Merge lại
echo "\n[STEP 1/3] Fine-tuning the LLM and merging adapters... (This may take a long time, especially on CPU)"
python src/01_finetune_and_merge.py
if [ $? -ne 0 ]; then
    echo "Model fine-tuning or merging failed. Exiting."
    exit 1
fi
echo "Fine-tuning and merging complete. Full model saved in 'models/fine_tuned_model/'"

# Bước 2: Đóng gói thành GGUF (Hướng dẫn thủ công)
echo "\n[STEP 2/3] PACKAGING MODEL TO GGUF (MANUAL STEP)"
echo "Please follow the instructions below to convert your model:"
echo "1. If you haven't, clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git"
echo "2. Build llama.cpp (e.g., run 'make' in the llama.cpp directory)."
echo "3. Creating output directory for GGUF model..."
mkdir -p models/gguf_model
echo "4. Run the conversion script (Note the updated script name):"
echo "   python llama.cpp/convert.py models/fine_tuned_model --outfile models/gguf_model/esg-detector.gguf"
echo "5. Quantize the model to run efficiently (e.g., Q4_K_M method):"
echo "   ./llama.cpp/quantize models/gguf_model/esg-detector.gguf models/gguf_model/esg-detector-q4_k_m.gguf q4_k_m"
echo "After this, load the 'esg-detector-q4_k_m.gguf' file in LM Studio."

# Bước 3: Chạy ứng dụng GUI
echo "\n[STEP 3/3] Running the Streamlit GUI application..."
echo "Make sure your GGUF model is loaded and the server is running in LM Studio."
streamlit run src/app.py

echo "\n======================================================"
echo "PIPELINE SCRIPT FINISHED."
echo "======================================================"
