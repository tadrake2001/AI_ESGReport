# src/02_finetune_and_merge.py
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import gc
import json
import re
from tqdm import tqdm
import ast

# --- Constants ---
# ĐÃ NÂNG CẤP LÊN PHIÊN BẢN 3B THEO YÊU CẦU
# CẢNH BÁO: YÊU CẦU GPU VỚI VRAM TỐI THIỂU 8-10GB
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# THAY ĐỔI: Cập nhật đường dẫn đến tệp dữ liệu mới, cân bằng và chất lượng hơn
DATASET_PATH = "data/dataset_esg_500_varied_2.0.json"
MERGED_MODEL_SAVE_PATH = "models/qwen2.5-esg-exaggeration-2.0"


def format_dataset(example):
    """
    Formats the dataset into the required chat template format for the model.
    The assistant's response is structured as a JSON string to ensure consistent,
    parsable output from the fine-tuned model.
    """
    output_json_string = json.dumps(example["output"], ensure_ascii=False, indent=2)
    input_text = "\n".join(example['input'])

    return {
        "text": f"<|im_start|>system\n"
                f"Bạn là một trợ lý chuyên gia về ESG. Phân tích văn bản được cung cấp để phát hiện các câu có dấu hiệu phóng đại. "
                f"Chỉ trả lời bằng một đối tượng JSON hợp lệ chứa nhãn 'label' và danh sách 'sentences'. "
                f"Mỗi đối tượng trong danh sách 'sentences' phải có 'sentence' và 'reason'.<|im_end|>\n"
                f"<|im_start|>user\n{example['instruction']}\n\n{input_text}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{output_json_string}\n<|im_end|>"
    }


def finetune_and_merge(train_dataset):
    """
    Main function to handle the fine-tuning and merging process.
    """
    print("--- Starting LLM Fine-tuning ---")

    # The SFTTrainer will handle formatting, so we don't need to map the dataset beforehand
    print("Dataset loaded. Formatting will be handled by the trainer.")

    use_gpu = torch.cuda.is_available()
    bnb_config = None
    device_map = "auto"
    torch_dtype = torch.float16
    optim_name = "paged_adamw_8bit"

    print("=" * 80)
    if use_gpu:
        print("✅ GPU DETECTED. The script will run on GPU.")
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print("   - Using 4-bit quantization (bitsandbytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        print("⚠️ WARNING: NO GPU DETECTED. The script will run on CPU.")
        print("   - This will be EXTREMELY SLOW and require a lot of RAM.")
        device_map = "cpu"
        torch_dtype = torch.float32
        optim_name = "adamw_torch"
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa"
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim=optim_name,
        logging_steps=20,
        learning_rate=5e-5,
        fp16=use_gpu,
        bf16=False,
        report_to="none",
    )

    # Define the formatting function for the trainer
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = format_dataset({
                "instruction": example['instruction'][i],
                "input": example['input'][i],
                "output": example['output'][i]
            })['text']
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func
    )

    print("Training...")
    trainer.train()
    print("Training complete.")

    final_adapter_path = os.path.join(training_args.output_dir, "final_adapter")
    trainer.save_model(final_adapter_path)
    print(f"Final adapter saved to {final_adapter_path}")

    print("--- Merging model and saving ---")
    del model
    del trainer
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype,
        return_dict=True,
        device_map="auto",
        trust_remote_code=True
    )

    merged_model = PeftModel.from_pretrained(base_model, final_adapter_path)
    merged_model = merged_model.merge_and_unload()

    os.makedirs(MERGED_MODEL_SAVE_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)
    print(f"Fine-tuned and merged model saved to {MERGED_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Data file not found at '{DATASET_PATH}'")
        print(f"Please place your '{os.path.basename(DATASET_PATH)}' file in the 'data/' directory.")
    else:
        # Load the full dataset to train on all data
        full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

        print(f"Loaded {len(full_dataset)} samples for training.")

        # Run the fine-tuning and merging process on the full dataset
        finetune_and_merge(full_dataset)

        # --- Tự động tắt PC sau khi chạy xong ---
        print("Quá trình huấn luyện và hợp nhất đã hoàn tất. Máy tính sẽ tự động tắt sau 10 giây...")
        import time
        time.sleep(10)
        if os.name == "nt":
            os.system("shutdown /s /t 1")
        else:
            os.system("shutdown -h now")
