import base64
from io import BytesIO
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler

app = FastAPI(title="API SERVER")

@app.get("/get-health")
def get_server_health():
    return {"status": "ok"}

def load_diffuser():
    image_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", use_safetensors=True, safety_checker=None).to("mps")
    image_model.scheduler = EulerDiscreteScheduler.from_config(image_model.scheduler.config)
    image_model.enable_attention_slicing()
    return image_model

img_model = load_diffuser()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=3001)