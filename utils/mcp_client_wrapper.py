from dotenv import load_dotenv
load_dotenv("../env.dev")

import asyncio
import json
from PIL import Image
import os
import io
import base64
from fastmcp import Client
from typing import List, Dict, Any, Union
import gradio as gr
from openai import AsyncOpenAI
from gradio.components.chatbot import ChatMessage
from fastmcp.client.client import CallToolResult

from .tools import tool_definition_list
from .mcp_utils import (
    call_image_generation_tool,
    call_image_describe_tool,
    call_get_forecast_tool,
    call_get_alerts_tool,
    call_get_multiply_tool,
    add_tool_response,
    add_image_tool_response
)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "")
OLLAMA_LLM_URL = os.getenv("OLLAMA_LLM_URL", "")

class MCPClientWrapper:
    def __init__(self):
        self.model_name = "bartowski/Qwen2.5-3B-Instruct-GGUF:Q5_K_S"
        self.mcp_client = Client(f"{MCP_SERVER_URL}/mcp")
        self.llm = AsyncOpenAI(
            base_url = f"{OLLAMA_LLM_URL}/v1",
            api_key='llama.cpp', # required, but unused
        )
        self.tools = tool_definition_list
        self.claude_messages = []
        self.result_messages = []
        self.claude_messages.append({
            "role": "system", 
            "content": 
"""
You're a chatbot assistant. Your task is to heed the user query and decide whether to use the functions such as: 'generate_image', 'describe_image', 'get_forecast', 'get_alerts' with their respective parameters or not.
Based on the the user query, decide if it is a conversation query or a functional tool request.
If the user's query are general, just response in a conversational manner.
If tools are needed, response with JSON format with the required parameters.
Use these tool definitions to help you identifying the tasks:
For tool 'generate_image', you must reponse with a JSON object with three key and value pairs representing three paramters: 'prompt', 'width' and 'height'.
For tool 'describe_image', you must response with a JSON object in the 'prompt' key with prompt representing the additional detail prompt for the image description as the parameter.
For tool 'get_alerts', you must response with a JSON object with a key and value pair representing the US state in the format of two-letter (e.g CA, NY) as parameter.
For tool 'get_forecast', if the latitude and longtitude are given by the user, use that and response with a JSON object representing two key and value pairs for 'latitude' and 'longtitude' parameters. If both of those are not provided, figure it out yourself.
For tool 'get_multiply', you must response with a JSON object with two key and value pairs representing the 'first_number' and the 'second_number' as parameters for the multiplication.
"""
        })
        
    async def check_connection(self):
        async with self.mcp_client:
            await self.mcp_client.ping()
            print("Server is reachable")
    
    async def _connect(self) -> str:
        async with self.mcp_client:
            self.mcp_client.session.__aenter__
            print(f"Client connected: {self.mcp_client.is_connected()}")

            # Make MCP calls within the context
            tools = await self.mcp_client.list_tools()
            print(f"Connected to MCP server. Available tools: {', '.join([t.name for t in tools])}")

    async def process_message(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]], img):
        # Initialize image_data to None
        image_data = None
        
        # Async generator to stream partial responses from _process_query
        async for partial_messages, partial_image_data in self._process_query(message, history, img):
            image_data = partial_image_data
            yield history + partial_messages, gr.Textbox(value=""), image_data

    async def _get_model_response_tool(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):
        response = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=message,
            tools=self.tools,
            tool_choice='auto'
        )
        return response
    
    async def _process_query(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]], img):
        self.result_messages = []
        
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")
            
            if role in ["user", "assistant", "system"]:
                self.claude_messages.append({"role": role, "content": content})
        
        # self.claude_messages.append({"role": "user", "content": message})
        
        print(f"[MESSAGE FIRST] - {self.claude_messages}")
        response = await self._get_model_response_tool(self.claude_messages, history=history)
        
        # Get the first choice from response
        choice = response.choices[0]
        print(f"[FIRST CHOICE] - {choice}")
        message = choice.message
        
        image_data = None
        
        # Handle regular text response
        if not message.tool_calls:
            self.result_messages.append({
                "role": "assistant",
                "content": message.content
            })
            print(f"[RESULT MES] - First res: {message.content}")
        else:
            # Handle tool calls like in your Anthropic code
            for tool_call in message.tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                print(f"[TOOL] - {tool_id} - {tool_name}")
                
                # Try to parse tool arguments
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"raw_args": tool_call.function.arguments}
                
                # Add initial tool use message
                self.result_messages.append({
                    "role": "assistant",
                    "content": f"I'll use the {tool_name} tool to help answer your question.",
                    "metadata": {
                        "title": f"Using tool: {tool_name}",
                        "log": f"Parameters: {json.dumps(tool_args, ensure_ascii=True)}",
                        "status": "pending",
                        "id": f"tool_call_{tool_name}"
                    }
                })
                
                # Add tool parameters message
                self.result_messages.append({
                    "role": "assistant",
                    "content": "```json\n" + json.dumps(tool_args, indent=2, ensure_ascii=True) + "\n```",
                    "metadata": {
                        "parent_id": f"tool_call_{tool_name}",
                        "id": f"params_{tool_name}",
                        "title": "Tool Parameters"
                    }
                })
                
                print(f"[RESULT MES] AFTER TOOL: {self.result_messages}")
                
                result = "Fail to get the server response"
                
                print(f"TOOL ARGS: {tool_args}")
                if tool_name == "generate_image":
                    result: CallToolResult | list[dict] | list = await call_image_generation_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                    # Extract image bytes from result
                    if result and isinstance(result, CallToolResult) and hasattr(result, "data"):
                        image_bytes = base64.b64decode(result.data)
                        # Convert bytes to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        image_data = image
                elif tool_name == "describe_image":
                    result: CallToolResult | str = await call_image_describe_tool(
                        file_byte=img,
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                elif tool_name == "get_forecast":
                    result: CallToolResult | list[dict] | str = await call_get_forecast_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                elif tool_name == "get_alerts":
                    result: CallToolResult | list[dict] | str = await call_get_alerts_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                elif tool_name == "get_multiply":
                    result: CallToolResult | list[dict] = await call_get_multiply_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                print(f"[RESULT MES] RESULT TOOL CALL MESSAGE: {self.result_messages}")
            
            # Get model to respond to tool output (second call)
            # Add tool results to messages
            self.claude_messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })
            
            # Retrieve the tool repsonse and add it in the LLM chat completion
            # Modify each tool response format if necessary.
            if tool_name == "generate_image":
                tool_response: dict = await add_image_tool_response(
                    result=tool_args.get("prompt", ""),
                    tool_id=tool_id,
                    tool_name=tool_name
                )
            elif tool_name == "describe_image":
                tool_response: dict = await add_tool_response(
                    tool_name=tool_name,
                    result=result,
                    tool_id=tool_id
                )
            elif tool_name == "get_multiply":
                tool_response: dict = await add_tool_response(
                    tool_name=tool_name,
                    result=result,
                    tool_id=tool_id
                )
            elif tool_name == "get_alerts" or tool_name == "get_forecast":
                tool_response: dict = await add_tool_response(
                    tool_name=tool_name,
                    result=result,
                    tool_id=tool_id
                )
                
            print(f"[TOOL RESPONSE] {tool_response}")
            self.claude_messages.append(tool_response)
        
        print(f"[FINAL_MESSAGE] GOT CLAUDE MESSAGE: {self.claude_messages}")
        # Get final response after tool use
        final_response = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=self.claude_messages,
            stream=True
        )
        
        partial_content = ""
        async for chunk in final_response:
            if chunk.choices[0].delta.content is not None:
                partial_content += chunk.choices[0].delta.content
                # Yield partial messages with updated content
                yield [{"role": "assistant", "content": partial_content}], image_data
