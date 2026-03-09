"""
FastAPI Server for the MCP Assistant.
Provides API endpoints for image description and audio transcription using local ML models.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("./env.dev")

import traceback
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM
from qwen_asr import Qwen3ASRModel
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from config import FASTAPI_HOST, FASTAPI_PORT


def load_asr_model():
    """Load the Qwen ASR model.
    
    Returns:
        Qwen3ASRModel instance
    """
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.float16,
        device_map="mps",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    return asr_model


def load_visual_llm():
    """Load the DeepSeek-VL visual language model.
    
    Returns:
        Tuple of (processor, model, tokenizer)
    """
    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16)
    vl_gpt = vl_gpt.to("mps").eval()
    
    return vl_chat_processor, vl_gpt, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads and unloads ML models."""
    print(f"LOADING ASR MODEL...")
    app.state.asr_model = load_asr_model()
    print(f"LOADED ASR MODEL.")
    
    # print(f"LOADING VLM MODEL...")
    # app.state.vl_chat_processor, app.state.vl_gpt, app.state.tokenizer = load_visual_llm()
    # print(f"LOADED VLM MODEL...")
    
    yield
    
    # Clean up the ML models and release the resources
    print(f"SHUTTING DOWN...")
    del app.state.asr_model
    del app.state.vl_chat_processor
    del app.state.vl_gpt
    del app.state.tokenizer


app = FastAPI(title="API SERVER", lifespan=lifespan)


@app.get("/get-health")
def get_server_health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/image/describe")
async def describe_image(request: Request, prompt: str, file_byte: str) -> dict:
    """Describe an uploaded image using DeepSeek-VL visual language model.
    
    Args:
        request: FastAPI request object
        prompt: Text prompt for image description
        file_byte: Base64 encoded image bytes
        
    Returns:
        Dictionary with status and description message
    """
    try:
        vl_chat_processor = request.app.state.vl_chat_processor
        vl_gpt = request.app.state.vl_gpt
        tokenizer = request.app.state.tokenizer
        
        # Prepare conversation with image placeholder
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>Describe this image with the detail: {prompt}.",
                "images": [file_byte]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        
        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)
        print(f"[DESCRIBE_IMAGE] GOT IMAGE")
        
        # Run image encoder to get embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response from model
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"[DESCRIBE_IMAGE] Done - Answer: {answer}")
        
        return {
            "status": "success",
            "message": answer,
        }
    
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][DESCRIBE_IMAGE] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error describing image: {str(e)}"
        }


@app.get("/audio/transcribe")
async def transcribe_audio(request: Request, prompt: str, file_path: str) -> dict:
    """Transcribe an audio file using Qwen ASR model.
    
    Args:
        request: FastAPI request object
        prompt: Text prompt for transcription
        file_path: Path to the audio file
        
    Returns:
        Dictionary with status and transcription text
    """
    asr_model = request.app.state.asr_model
    try:
        result = asr_model.transcribe(
            audio=file_path,
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
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)

