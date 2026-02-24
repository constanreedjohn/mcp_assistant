import torch
import time
from qwen_asr import Qwen3ASRModel

print(f"=== LOADING BASE MODEL...")
start_load_2 = time.perf_counter()
model_2 = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.float16,
    device_map="mps",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
)
end_load_2 = time.perf_counter()

print(f"=== LOADED BASE MODEL...")

start_2 = time.perf_counter()
results_2 = model_2.transcribe(
    # audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    audio="../input_audio.wav",
    language=None, # set "English" to force the language
)
end_2 = time.perf_counter()

print(f"============\nRESULT_2\n")
print(results_2[0].language)
print(results_2[0].text)