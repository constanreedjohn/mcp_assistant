"""
Centralized configuration for the MCP Assistant application.
Loads environment variables and provides default values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / "env.dev"
load_dotenv(env_path)

# ============================================================================
# Server Configuration
# ============================================================================

# MCP Server URL
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:5001")

# Llama.cpp LLM URL
LLAMACPP_LLM_URL: str = os.getenv("LLAMACPP_LLM_URL", "http://127.0.0.1:4001")

# SLM URL (for intent classification)
SLM_URL: str = os.getenv("SLM_URL", "http://127.0.0.1:6001")

# Image Generation URL
IMAGE_GEN_URL: str = os.getenv("IMAGE_GEN_URL", "http://127.0.0.1:3001")

# ============================================================================
# Model Configuration
# ============================================================================

# Main LLM model (for chat + tool calls)
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "bartowski/Qwen_Qwen3.5-2B-GGUF:Q4_1")

# SLM model name (for intent classification)
SLM_MODEL_NAME: str = os.getenv("SLM_MODEL_NAME", "bartowski/Qwen_Qwen3.5-0.8B-GGUF:Q5_K_M")

# Path to SLM model file (for subprocess-based classifier)
SLM_MODEL_PATH: str = os.getenv(
    "SLM_MODEL_PATH", 
    str(Path(__file__).parent / "models" / "bartowski_Qwen_Qwen3.5-0.8B-GGUF_Qwen_Qwen3.5-0.8B-Q5_K_M.gguf")
)

# ============================================================================
# Audio Configuration
# ============================================================================

SAMPLE_RATE: int = 16000
MAX_ASR_INPUT_SECONDS: int = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS: int = 180
MIN_ASR_INPUT_SECONDS: float = 0.5

# ============================================================================
# API Configuration
# ============================================================================

NWS_API_BASE: str = "https://api.weather.gov"
USER_AGENT: str = "weather-app/1.0"

# ============================================================================
# File Paths
# ============================================================================

# Temporary input files (used by MCP tools)
INPUT_AUDIO_PATH: str = str(Path(__file__).parent.parent / "input_audio.wav")
INPUT_IMAGE_PATH: str = str(Path(__file__).parent.parent / "input.jpg")

# ============================================================================
# Application Settings
# ============================================================================

# Gradio server settings
GRADIO_HOST: str = "0.0.0.0"
GRADIO_PORT: int = 7860

# FastAPI server settings
FASTAPI_HOST: str = "0.0.0.0"
FASTAPI_PORT: int = 3001

# MCP server settings
MCP_HOST: str = "0.0.0.0"
MCP_PORT: int = 5001

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

