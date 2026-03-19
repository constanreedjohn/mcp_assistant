import base64
import numpy as np
from io import BytesIO
from fastapi import FastAPI, Request
import traceback
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM
from qwen_asr import Qwen3ASRModel

def load_asr():
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.float16,
        device_map="mps",
        max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )
    return asr_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"LOADING ASR MODEL...")
    app.state.asr_model = load_asr()
    print(f"LOADED ASR MODEL.")
    
    yield
    
    # Clean up the ML models and release the resources
    print(f"SHUTTING DOWN...")
    # Optional: explicitly delete models to free memory
    del app.state.asr_model
    del app.state.vl_chat_processor
    del app.state.vl_gpt
    del app.state.tokenizer
    
app = FastAPI(title="API SERVER", lifespan=lifespan)

@app.get("/get-health")
def get_server_health():
    return {"status": "ok"}

@app.get("/audio/transcribe")
async def transcribe_audio(request: Request, prompt: str, file_path: str):
    asr_model: Qwen3ASRModel = request.app.state.asr_model
    try:
        input_data = "../input_audio.wav"
        result = asr_model.transcribe(
            audio=input_data,
            language=None,
        )
        
        return {
            "status": "success",
            "message": result[0].text,
        }
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][TRANSCRIBE AUDIO] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error transcribing audio: {str(e)}"
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=3001)