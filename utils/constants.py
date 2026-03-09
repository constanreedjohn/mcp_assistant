"""
Constants and shared configuration for the MCP Assistant.
"""
from pathlib import Path

# ============================================================================
# Audio Configuration
# ============================================================================

SAMPLE_RATE: int = 16000
MAX_ASR_INPUT_SECONDS: int = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS: int = 180
MIN_ASR_INPUT_SECONDS: float = 0.5

# ============================================================================
# Weather API Configuration
# ============================================================================

NWS_API_BASE: str = "https://api.weather.gov"
USER_AGENT: str = "weather-app/1.0"

# ============================================================================
# Model Configuration
# ============================================================================

DEFAULT_LLM_MODEL: str = "bartowski/Qwen_Qwen3.5-2B-GGUF:Q4_1"
DEFAULT_SLM_MODEL: str = "bartowski/Qwen_Qwen3.5-0.8B-GGUF:Q5_K_M"

# ============================================================================
# Tool Names
# ============================================================================

TOOL_TRANSCRIBE_AUDIO: str = "transcribe_audio"
TOOL_DESCRIBE_IMAGE: str = "describe_image"
TOOL_GET_ALERTS: str = "get_alerts"
TOOL_GET_FORECAST: str = "get_forecast"
TOOL_GET_MULTIPLY: str = "get_multiply"
TOOL_GENERATE_IMAGE: str = "generate_image"

# ============================================================================
# Default Paths
# ============================================================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default input files (relative to project root)
DEFAULT_INPUT_AUDIO: str = str(PROJECT_ROOT / "input_audio.wav")
DEFAULT_INPUT_IMAGE: str = str(PROJECT_ROOT / "input.jpg")

# ============================================================================
# System Prompts
# ============================================================================

SYSTEM_PROMPT = """You're a chatbot assistant. Your task is to heed the user query and decide whether to use the functions such as: 'transcribe_audio', 'describe_image', 'get_forecast', 'get_alerts' with their respective parameters or not.
Based on the the user query, decide if it is a conversation query or a functional tool request.
If the user's query are general, just response in a conversational manner.
If tools are needed, response with JSON format with the required parameters.
Use these tool definitions to help you identifying the tasks:
* For tool 'transcribe_audio', you must reponse with a JSON object in the 'prompt' key with prompt representing the additional detail prompt for the audio transcription as the parameter.
* For tool 'describe_image', you must response with a JSON object in the 'prompt' key with prompt representing the additional detail prompt for the image description as the parameter.
* For tool 'get_alerts', you must response with a JSON object with a key and value pair representing the US state in the format of two-letter (e.g CA, NY) as parameter.
* For tool 'get_forecast', if the latitude and longtitude are given by the user, use that and response with a JSON object representing two key and value pairs for 'latitude' and 'longtitude' parameters. If both of those are not provided, figure it out yourself.
* For tool 'get_multiply', you must response with a JSON object with two key and value pairs representing the 'first_number' and the 'second_number' as parameters for the multiplication.
"""

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier, your task is to reponse with either 'tool' or 'chat' based on the context of the QUERY. Use the RULES to help you in generating the response.        
-------------
RULES:
* Based on the context of the QUERY, response with only 'tool' or 'chat'.
* If the QUERY contains relevant information relate to these keywords: ['transcribe audio', 'provide weather forecast', 'provide weather alerts'], response 'tool'.
* If the QUERY does not contain relevant information in those keywords or just a general conversation knowledge, then response 'chat'.
-------------
QUERY: {query_message}
"""

