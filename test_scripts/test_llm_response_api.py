from dotenv import load_dotenv
load_dotenv("../env.dev")

import asyncio
import json
from PIL import Image
import subprocess
import os
import io
import base64
import numpy as np
import soundfile as sf
from fastmcp import Client
from typing import List, Dict, Any, Union, Tuple
import gradio as gr
from openai import AsyncOpenAI, OpenAI
from gradio.components.chatbot import ChatMessage
from fastmcp.client.client import CallToolResult
from typing import List, Dict, Any, Union

import asyncio

class IntentClassifier:
    def __init__(self):
        self.process = None

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "/opt/homebrew/bin/llama-completion",
            "-m", "/Users/codelink/Desktop/Projects/misc/mcp_test/mcp_assistant/models/bartowski_Qwen_Qwen3.5-0.8B-GGUF_Qwen_Qwen3.5-0.8B-Q5_K_M.gguf",
            "--reasoning-budget", "0",
            "-n", "5",
            "--temp", "0",
            "-i",          # interactive mode — keeps process alive, reads from stdin
            "--no-display-prompt",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def classify_intent(self, user_q: str) -> str:
        if self.process is None or self.process.returncode is not None:
            await self.start()  # restart if dead

        prompt = f"""You are an intent classifier, your task is to reponse with either 'tool' or 'chat' based on the context of the QUERY. Use the RULES to help you in generating the response.        
-------------
RULES:
* Based on the context of the QUERY, response with only 'tool' or 'chat'.
* If the QUERY contains context relevant to the definitions from the CONTEXT_DEFINITION, response 'tool'.
* If the QUERY contains context unrelated from the CONTEXT_DEFINITION or just a general conversation knowledge, then response 'chat'.
-------------
CONTEXT_DEFINITION
* When the QUERY is asking to transcribe an audio file from an uploaded file.
* When the QUERY is asking to describe an image.
* When the QUERY is asking to check the weather alerts.
* When the QUERY is asking to check the weather forecast.
* When the QUERY is asking for a multiplication


    QUERY: {user_q}
"""
        self.process.stdin.write((prompt + "\n").encode())
        await self.process.stdin.drain()

        output = (await self.process.stdout.readline()).decode().strip().lower()
        print(f"OUTPUT: {output}")
        # return "tool" if "tool" in output else "chat"

    async def stop(self):
        if self.process:
            self.process.stdin.close()
            await self.process.wait()

# from .logging_utils import logger
# from .utils import _read_wav_from_bytes
# from .mcp_utils import (
#     call_image_generation_tool,
#     call_image_describe_tool,
#     call_get_forecast_tool,
#     call_get_alerts_tool,
#     call_get_multiply_tool,
#     call_transcribe_audio_tool,
#     add_tool_response,
#     add_image_tool_response
# )

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
LLAMACPP_LLM_URL = os.getenv("LLAMACPP_LLM_URL", "")

class MCPClientWrapper:
    def __init__(self):
        self.model_name = "bartowski/Qwen2.5-3B-Instruct-GGUF:Q5_K_S"
        self.mcp_client = Client(f"{MCP_SERVER_URL}/mcp")
        self.llm = OpenAI(
            base_url = f"{LLAMACPP_LLM_URL}/v1",
            api_key='llama.cpp', # required, but unused
        )
        self.tools = "tool_definition_list"
        self.result_messages = []
        self.sys_prompt = """
You're a chatbot assistant. Your task is to heed the user query and decide whether to use the functions such as: 'transcribe_audio', 'describe_image', 'get_forecast', 'get_alerts' with their respective parameters or not.
Based on the the user query, decide if it is a conversation query or a functional tool request.
If the user's query are general, just response in a conversational manner.
If tools are needed, response with JSON format with the required parameters.
Use these tool definitions to help you identifying the tasks:
For tool 'transcribe_audio', you must reponse with a JSON object in the 'prompt' key with prompt representing the additional detail prompt for the audio transcription as the parameter.
For tool 'describe_image', you must response with a JSON object in the 'prompt' key with prompt representing the additional detail prompt for the image description as the parameter.
For tool 'get_alerts', you must response with a JSON object with a key and value pair representing the US state in the format of two-letter (e.g CA, NY) as parameter.
For tool 'get_forecast', if the latitude and longtitude are given by the user, use that and response with a JSON object representing two key and value pairs for 'latitude' and 'longtitude' parameters. If both of those are not provided, figure it out yourself.
For tool 'get_multiply', you must response with a JSON object with two key and value pairs representing the 'first_number' and the 'second_number' as parameters for the multiplication.
"""
        
    async def _get_model_completion_tool(self, message: dict):
        self.result_messages += [
            {
                'role': 'system',
                'content': self.sys_prompt
            }
        ]
        self.result_messages.append(message)
        response = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=self.result_messages,
            tool_choice='auto',
            tools='tool_definition_list',
        )
        print(response)

async def main():
    classifier = IntentClassifier()
    await classifier.start()
    
    import time
    start = time.perf_counter()
    print(await classifier.classify_intent("what's the weather today?"))
    end = time.perf_counter() - start
    # print(temp)
    # print(f"{end:.2f}")
    
    await classifier.start()
    
if __name__ == "__main__":
    asyncio.run(main())