from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/ubuntu/デスクトップ/Python/unsloth_system/data/models/logical_9b",
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

FastLanguageModel.for_inference(model)

model.save_pretrained_gguf("logical_model_9b", tokenizer, quantization_method = "q4_k_m")
