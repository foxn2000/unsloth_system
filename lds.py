from datasets import load_dataset

EOS_TOKEN = "<eos>"

default_prompt = """以下は、タスクを説明する命令と、さらなる文脈を提供する入力の組み合わせである。要求を適切に完了する応答を書きなさい。

### Instruction:
あなたは世界最高レベルにある価値投資の達人です。長期的な視点で、安全域のある優良企業を探し出し、詳細な分析に基づいた投資判断と根拠を示すことが得意です。

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    # instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instructions,input, output in zip(instructions,inputs, outputs):
        # EOSトークンを追加
        text = default_prompt.format(instructions,input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def return_default_prompt():
    return default_prompt

def dataset_load():
    # DATA_PATH = "/home/ubuntu/デスクトップ/python/make-datasets/in-foxhound-ja.jsonl"
    # dataset = load_dataset("json", data_files=DATA_PATH,split="train")
    dataset = load_dataset("DataPilot/Generated-dataset-by-deepseek-v2.5", split="train")
    # データセットをフォーマット
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
