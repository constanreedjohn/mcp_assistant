import base64
import numpy as np
from io import BytesIO
from fastapi import FastAPI, Request
import traceback
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM
from qwen_asr import Qwen3ASRModel
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def load_asr():
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.float16,
        device_map="mps",
        max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
        max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
    )
    return asr_model

def load_diffuser():
    image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", use_safetensors=True, safety_checker=None).to("mps")
    image_model.scheduler = EulerDiscreteScheduler.from_config(image_model.scheduler.config)
    image_model.enable_attention_slicing()
    return image_model

def load_visual_llm():
    # Initialize DeepSeek-VL model and processor
    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16)
    vl_gpt = vl_gpt.to("mps").eval()
    return vl_chat_processor, vl_gpt, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"LOADING ASR MODEL...")
    app.state.asr_model = load_asr()
    print(f"LOADED ASR MODEL.")
    
    # print(f"LOADING STABLEDIF MODEL...")
    # app.state.img_model = load_diffuser()
    # print(f"LOADED STABLEDIF MODEL...")
    
    print(f"LOADING VLM MODEL...")
    app.state.vl_chat_processor, app.state.vl_gpt, app.state.tokenizer = load_visual_llm()
    print(f"LOADED VLM MODEL...")
    
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

@app.get("/image/generate")
async def generate_image(prompt: str, width: int = 512, height: int = 512) -> dict:
    """Generate an image using local model.
    
    Args:
        prompt: Text prompt describing the image to generate
        width: Image width (default: 512)
        height: Image height (default: 512)
    """
    print(f"[SERVER][GEN_LOCAL_IMAGE] Triggered")
    try:
        # Generate the image
        image = img_model(prompt, num_inference_steps=25).images[0]
        
        # Save image to file (optional, you can keep or remove this)
        image.save("./result.png")
        
        # Convert image to bytes for the API response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        # Encode as base64 for JSON response
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        print(f"[SERVER][GEN_LOCAL_IMAGE] Done")
        return {
            "status": "success",
            "message": f"Generated image for prompt: {prompt}",
            "image_bytes": img_base64
        }
        
    except Exception as e:
        print(f"[SERVER][GEN_LOCAL_IMAGE] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating image: {str(e)}",
            "image_bytes": None
        }

@app.get("/image/describe")
async def describe_image(request: Request, prompt: str, file_byte: str) -> dict:
    """Describe an uploaded image using DeepSeek-VL visual language model."""
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
        print(f"[SERVER][GEN_LOCAL_IMAGE] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error describing image: {str(e)}"
        }

@app.get("/audio/transcribe")
async def transcribe_audio(request: Request, audio_data: list, sample_rate: int):
    asr_model: Qwen3ASRModel = request.app.state.asr_model
    try:
        input_data = (audio_data, sample_rate)
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