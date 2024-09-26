## main.txtのクリーニング版
import torch
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from lds import dataset_load

model_path = "data/lora_adapter/"
model_name = "lora_model"

# 最大シーケンス長を設定します。RoPEスケーリングが内部で自動的にサポートされるため、任意の値を選択できます。
max_seq_length =8192

# データ型を指定します。自動検出の場合はNone、Tesla T4/V100の場合はFloat16、Ampere+の場合はBfloat16を指定します。
dtype = None

# メモリ使用量を削減するために4ビット量子化を使用するかどうかを指定します。
load_in_4bit = True

# 事前量子化された4ビットモデルのリストです。これらのモデルはダウンロードが4倍速く、OOMエラーが発生しません。
# 詳細なモデルリストは https://huggingface.co/unsloth を参照してください。
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
]

# FastLanguageModelとトークナイザーをロードします。
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map='auto'
)

# PEFT (Parameter-Efficient Fine-Tuning) を使用してモデルを微調整します。
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRAのランク。大きいほど表現力が高くなりますが、メモリ使用量も増加します。
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # LoRAを適用するモジュール
    lora_alpha=16,  # LoRAのスケーリング係数
    lora_dropout=0,  # LoRAのドロップアウト率
    bias="none",  # バイアスの適用方法
    use_gradient_checkpointing="unsloth",  # 勾配チェックポイントを使用するかどうか
    random_state=3407,  # 乱数シード
    use_rslora=False,  # Rank Stabilized LoRAを使用するかどうか
    loftq_config=None,  # LoftQの設定
)

dataset = dataset_load()

# SFTTrainerを使用してモデルをファインチューニング
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # データセット処理のプロセス数
    packing=False,  # シーケンスパッキングを使用するかどうか
    args=TrainingArguments(
        per_device_train_batch_size=2,  # デバイスごとのバッチサイズ
        gradient_accumulation_steps=4,  # 勾配蓄積ステップ数
        warmup_steps=5,  # ウォームアップステップ数
        max_steps=60,  # 最大トレーニングステップ数
        learning_rate=2e-4,  # 学習率
        fp16=not is_bfloat16_supported(),  # FP16を使用するかどうか
        bf16=is_bfloat16_supported(),  # BF16を使用するかどうか
        logging_steps=1,  # ログ出力ステップ数
        optim="adamw_8bit",  # オプティマイザー
        weight_decay=0.007,  # 重み減衰
        lr_scheduler_type="linear",  # 学習率スケジューラー
        seed=3407,  # 乱数シード
        output_dir="outputs",  # 出力ディレクトリ
    ),
)

# GPUのメモリ使用量を表示
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# モデルのトレーニング
trainer_stats = trainer.train()

model.save_pretrained(model_path+model_name)
tokenizer.save_pretrained(model_path+model_name)