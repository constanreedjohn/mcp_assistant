import asyncio
import json
from PIL import Image
import io
import base64
from fastmcp import Client
from typing import List, Dict, Any, Union
import gradio as gr
from openai import AsyncOpenAI
from gradio.components.chatbot import ChatMessage

from .tools import tool_definition_list
from .mcp_utils import (
    call_image_generation_tool,
    call_get_forecast_tool,
    call_get_alerts_tool,
    call_get_multiply_tool,
    add_tool_response,
    add_image_tool_response
)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

class MCPClientWrapper:
    def __init__(self):
        self.model_name = "llama3.1:8b"
        self.mcp_client = Client("http://127.0.0.1:5001/mcp")
        self.llm = AsyncOpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key='ollama', # required, but unused
        )
        self.tools = tool_definition_list
        self.claude_messages = []
        self.result_messages = []
        
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

    def process_message(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]) -> tuple:
        # Initialize image_data to None
        image_data = None
        
        # Run the async _process_query and get the result messages
        new_messages, image_data = loop.run_until_complete(self._process_query(message, history))
        
        # Check if any of the result messages contain image bytes
        for msg in new_messages:
            # The image bytes are in the tool call result, check for 'metadata' and 'title' keys
            if "metadata" in msg and msg["metadata"].get("title", "").startswith("Tool Result for generate_image"):
                # The actual image bytes are in the previous message content (result list)
                # We can try to find the image bytes in the result list returned by call_image_generation_tool
                # But since _process_query does not return the raw result, we need to extract from the messages
                # Instead, we will modify _process_query to return the image bytes as well (done below)
                pass
        
        # Return updated history, empty string for textbox, and image_data for display_image
        return history + [{"role": "user", "content": message}] + new_messages, gr.Textbox(value=""), image_data

    async def _get_model_response(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]], tool=False):
        if tool:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=message,
                tools=self.tools,
                tool_choice='auto'
            )
        else:
            response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=message,
            )
        return response
    
    async def _process_query(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):

        self.result_messages = []
        self.claude_messages.append({
            "role": "system", 
            "content": 
"""
You're a chatbot assistant. Your task is to heed the user instruction and decide whether to use the functions such as: 'generate_image', 'get_forecast', 'get_alerts' with their respective parameters or not.
If the user's question are general, just response with conversational manner.
If function are needed, response with JSON format with the required parameters.
Use these function definitions to help you identifying the tasks:
For function 'generate_image', you must reponse with a JSON object with three key and value pairs representing three paramters: 'prompt', 'width' and 'height'.
For function 'get_alerts', you must response with a JSON object with a key and value pair representing the US state in the format of two-letter (e.g CA, NY) as parameter.
For function 'get_forecast', if the latitude and longtitude are given by the user, use that and response with a JSON object representing two key and value pairs for 'latitude' and 'longtitude' parameters. If both of those are provided, figure it out yourself.
For function 'get_multiply', you must response with a JSON object with two key and value pairs representing the 'first_number' and the 'second_number' as parameters for the multiplication.
"""
        })
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")
            
            if role in ["user", "assistant", "system"]:
                self.claude_messages.append({"role": role, "content": content})
        
        self.claude_messages.append({"role": "user", "content": message})
        
        print(f"[MESSAGE FIRST] - {self.claude_messages}")
        response = await self._get_model_response(self.claude_messages, history=history, tool=True)
        
        # Get the first choice from response
        choice = response.choices[0]
        print(f"[FIRST CHOICE] - {choice}")
        message = choice.message
        
        # Initialize image_data to None
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
                    result: list[dict] | list = await call_image_generation_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                    # Extract image bytes from result
                    if result and isinstance(result, list) and hasattr(result[0], "data"):
                        image_bytes = base64.b64decode(result[0].data)
                        # Convert bytes to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        image_data = image
                elif tool_name == "get_forecast":
                    result: list[dict] = await call_get_forecast_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                elif tool_name == "get_alerts":
                    result: list[dict] = await call_get_alerts_tool(
                        mcp_client=self.mcp_client,
                        tool_args=tool_args,
                        result_messages=self.result_messages,
                        tool_name=tool_name
                    )
                elif tool_name == "get_multiply":
                    result: list[dict] = await call_get_multiply_tool(
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
                tool_response = add_image_tool_response(
                    result=tool_args.get("prompt", ""),
                    tool_id=tool_id
                )
            elif tool_name == "get_multiply":
                tool_response = add_tool_response(
                    result=result,
                    tool_id=tool_id
                )
            elif tool_name == "get_alerts" or tool_name == "get_forecast":
                tool_response = add_tool_response(
                    result=result,
                    tool_id=tool_id
                )
                
            print(f"[TOOL RESPONSE] {tool_response}")
            self.claude_messages.append(tool_response)
            
            # Get final response after tool use
            final_response = await self._get_model_response(
                message=self.claude_messages,
                history=history,
                tool=False
            )
            print(f"[RESULT MES] - Final response: {final_response}")
            
            # Add final response
            final_content = final_response.choices[0].message.content
            if final_content:
                self.result_messages.append({
                    "role": "assistant",
                    "content": final_content
                })

        return self.result_messages, image_data
    