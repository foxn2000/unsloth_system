## ライブラリ
import torch
import os
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_dataset import dataset_load,default_prompt

def lora_to_model(lora_name, model_name):

    lora_directory = "data/lora_adapter/"
    model_directory = "data/models/"

    peft_name = lora_directory+lora_name   #学習済みadapter_config.jsonのパス指定
    output_dir = model_directory+model_name  #マージモデルの出力先

    # PEFT(LoRA)の指定
    peft_config = PeftConfig.from_pretrained(peft_name)
    # ベースモデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )
    # Rinnaのトークナイザーでは、「use_fast=False」も必要になる
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,use_fast=False)
    # PEFT(LoRA)の読み込み
    model = PeftModel.from_pretrained(model, peft_name)
    # マージモデル作成
    merged_model = model.merge_and_unload()
    # 出力
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saving to {output_dir}")

def run_model(user_input, model_name):
    model_drectory = "data/models/"

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # モデルとトークナイザーをロード
    model_name = model_drectory+model_name  # 好きなモデル名に変更可能
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16) # デバイスを自動的に選択

    default_prompt = default_prompt()

    # 推論を実行
    inputs = tokenizer(
        [
            default_prompt.format(
                user_input,  # 入力
                "",  # 出力 - 生成の場合は空白にします
            )
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    return (tokenizer.batch_decode(outputs, skip_special_tokens=True))
