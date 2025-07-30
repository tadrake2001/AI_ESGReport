import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import re
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import ast

# --- Constants ---
# Path to the fine-tuned and merged model
MERGED_MODEL_SAVE_PATH = "models/qwen2.5-esg-exaggeration-2.0"
# Path to the new test dataset provided by the user
TEST_DATASET_PATH = "data/dataset_esg_500_varied_2.0.json"


def evaluate_model(model_path, eval_dataset):
    """
    Evaluates the fine-tuned model, provides a detailed classification report,
    confusion matrix, and exports lists of incorrect predictions.
    """
    print("\n--- Starting Model Evaluation ---")

    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        trust_remote_code=True,
        # attn_implementation="sdpa"  # TĂNG TỐC: Sử dụng SDPA để tăng tốc độ suy luận
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    y_true = []  # Ground truth labels for all sentences
    y_pred = []  # Predicted labels for all sentences
    total_exact_matches = 0

    # NÂNG CẤP: Khởi tạo danh sách để lưu các câu bị dự đoán sai
    false_positives_list = []
    false_negatives_list = []

    system_prompt = (
        "Bạn là một trợ lý chuyên gia về ESG. Phân tích văn bản được cung cấp để phát hiện các câu có dấu hiệu phóng đại. "
        "Chỉ trả lời bằng một đối tượng JSON hợp lệ chứa nhãn 'label' và danh sách 'sentences'. "
        "Mỗi đối tượng trong danh sách 'sentences' phải có 'sentence' và 'reason'."
    )

    for example in tqdm(eval_dataset, desc="Evaluating"):
        instruction = example.get('instruction', '')
        input_sentences = example.get('input', [])
        ground_truth_output = example.get('output', {"sentences": []})

        input_text = "\n".join(input_sentences)
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n\n{input_text}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        output_sequences = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=False)

        try:
            assistant_part = response_text.split("<|im_start|>assistant\n")[1]
            assistant_part = assistant_part.replace("<|im_end|>", "").strip()
        except IndexError:
            assistant_part = ""

        # SỬA LỖI: Cải tiến logic xử lý JSON/dictionary string
        predicted_exaggerated_sents = set()
        parsing_failed = False

        try:
            json_start = assistant_part.find('{')
            json_end = assistant_part.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                dict_str = assistant_part[json_start:json_end]
                # Cố gắng phân tích chuỗi bằng ast.literal_eval (an toàn hơn json.loads cho đầu vào không đáng tin cậy)
                parsed_dict = ast.literal_eval(dict_str)
                predicted_exaggerated_sents = {s.get('sentence', '').strip() for s in parsed_dict.get('sentences', [])}
            else:
                # Không tìm thấy đối tượng JSON nào trong phản hồi
                parsing_failed = True
        except (ValueError, SyntaxError):
            # Lỗi cú pháp khi phân tích, chuyển sang phương pháp dự phòng
            parsing_failed = True

        if parsing_failed:
            print(
                f"\nWarning: Could not parse output. Falling back to sentence matching for:\n---\n{assistant_part}\n---")
            # Phương pháp dự phòng: Kiểm tra xem câu nào trong input có xuất hiện trong output
            for sent in input_sentences:
                # Kiểm tra xem câu có được trích dẫn trong output không
                if f"'{sent.strip()}'" in assistant_part or f'"{sent.strip()}"' in assistant_part:
                    predicted_exaggerated_sents.add(sent.strip())

        true_exaggerated_sents = {s.get('sentence', '').strip() for s in ground_truth_output.get('sentences', [])}

        predicted_exaggerated_sents.discard('')
        true_exaggerated_sents.discard('')

        # Generate labels for each sentence in the input
        for sentence in input_sentences:
            clean_sentence = sentence.strip()
            is_true_exaggerated = 1 if clean_sentence in true_exaggerated_sents else 0
            is_pred_exaggerated = 1 if clean_sentence in predicted_exaggerated_sents else 0
            y_true.append(is_true_exaggerated)
            y_pred.append(is_pred_exaggerated)

            # NÂNG CẤP: Kiểm tra và lưu lại các lỗi sai
            if is_pred_exaggerated == 1 and is_true_exaggerated == 0:
                false_positives_list.append(clean_sentence)
            elif is_pred_exaggerated == 0 and is_true_exaggerated == 1:
                false_negatives_list.append(clean_sentence)

        if predicted_exaggerated_sents == true_exaggerated_sents:
            total_exact_matches += 1

    # --- Display Results ---
    print("\n" + "=" * 50)
    print(" " * 15 + "EVALUATION RESULTS")
    print("=" * 50)

    # Table 1: Classification Report
    print("\nBảng 1: Kết quả phân loại câu phóng đại\n" + "-" * 50)
    target_names = ['Không phóng đại (0)', 'Phóng đại (1)']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)

    # Table 2: Confusion Matrix
    print("\nBảng 2: Ma trận nhầm lẫn\n" + "-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=['Thực tế: Không phóng đại', 'Thực tế: Phóng đại'],
                         columns=['Dự đoán: Không phóng đại', 'Dự đoán: Phóng đại'])
    print(cm_df)

    # THÊM MỚI: Thêm phần diễn giải chi tiết cho ma trận nhầm lẫn
    print("\nDiễn giải:")
    print(f"- True Negative (TN): {cm[0][0]} câu 'Không phóng đại' được dự đoán đúng.")
    print(f"- False Positive (FP): {cm[0][1]} câu 'Không phóng đại' bị dự đoán sai là 'Phóng đại'.")
    print(f"- False Negative (FN): {cm[1][0]} câu 'Phóng đại' bị bỏ lỡ (dự đoán sai là 'Không phóng đại').")
    print(f"- True Positive (TP): {cm[1][1]} câu 'Phóng đại' được dự đoán đúng.")

    accuracy_overall = (cm[0][0] + cm[1][1]) / len(y_true) if y_true else 0
    accuracy_exact_match = total_exact_matches / len(eval_dataset) if eval_dataset else 0
    print("\n" + "-" * 50)
    print(f"Độ chính xác tổng thể (từng câu): {accuracy_overall:.4f}")
    print(f"Độ chính xác khớp hoàn toàn (theo nhóm): {accuracy_exact_match:.4f}")
    print("=" * 50)

    # NÂNG CẤP: Hiển thị và lưu các câu bị sai
    if false_positives_list:
        print(f"\n--- Phân tích Lỗi sai: {len(false_positives_list)} False Positives ---")
        print("(Những câu bình thường bị dự đoán sai là 'phóng đại')")
        fp_df = pd.DataFrame(false_positives_list, columns=["Câu"])
        print(fp_df)
        fp_df.to_csv("false_positives.csv", index=False)
        print("\n-> Đã lưu danh sách False Positives vào tệp 'false_positives.csv'")

    if false_negatives_list:
        print(f"\n--- Phân tích Lỗi sai: {len(false_negatives_list)} False Negatives ---")
        print("(Những câu phóng đại thực sự đã bị bỏ lỡ)")
        fn_df = pd.DataFrame(false_negatives_list, columns=["Câu"])
        print(fn_df)
        fn_df.to_csv("false_negatives.csv", index=False)
        print("\n-> Đã lưu danh sách False Negatives vào tệp 'false_negatives.csv'")


if __name__ == "__main__":
    if not os.path.exists(MERGED_MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at '{MERGED_MODEL_SAVE_PATH}'")
        print("Please run the fine-tuning script first to generate the model.")
    elif not os.path.exists(TEST_DATASET_PATH):
        print(f"ERROR: Test data file not found at '{TEST_DATASET_PATH}'")
        print("Please make sure 'esg_exaggeration_test_set.json' is in the 'data/' directory.")
    else:
        test_dataset = load_dataset("json", data_files=TEST_DATASET_PATH, split="train")
        print(f"Loaded {len(test_dataset)} samples from {TEST_DATASET_PATH}")

        evaluate_model(MERGED_MODEL_SAVE_PATH, test_dataset)
