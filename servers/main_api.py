import base64
from io import BytesIO
from fastapi import FastAPI
import traceback
import torch
from transformers import AutoModelForCausalLM

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

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

img_model = load_diffuser()
vl_chat_processor, vl_gpt, tokenizer = load_visual_llm()

app = FastAPI(title="API SERVER")

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
        image = img_model(prompt).images[0]
        
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
async def describe_image(prompt: str, file_byte: str) -> dict:
    """Describe an uploaded image using DeepSeek-VL visual language model."""
    try:
        # Convert to base64
        # image_base64 = base64.b64encode(bytes(file_byte)).decode('utf-8')
        # mime_type = "image/jpeg"
        # image = f"data:{mime_type};base64,{image_base64}"
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=3001)