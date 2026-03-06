"""
MCP Client Wrapper for the MCP Assistant.
Handles communication with the MCP server and LLM integration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import time
from typing import List, Dict, Any, Union, Tuple
from PIL import Image

import soundfile as sf
from fastmcp import Client
from openai import AsyncOpenAI
from gradio.components.chatbot import ChatMessage
from fastmcp.client.client import CallToolResult

from .tools import tool_definition_list
from .logging_utils import logger
from .audio_utils import read_wav_from_bytes
from .constants import (
    SYSTEM_PROMPT, 
    INTENT_CLASSIFIER_PROMPT,
    DEFAULT_LLM_MODEL,
    DEFAULT_SLM_MODEL,
)
from config import MCP_SERVER_URL, LLAMACPP_LLM_URL, SLM_URL
from .mcp_utils import (
    call_image_generation_tool,
    call_image_describe_tool,
    call_get_forecast_tool,
    call_get_alerts_tool,
    call_get_multiply_tool,
    call_transcribe_audio_tool,
    add_tool_response,
    add_image_tool_response,
)


class SLMClientWrapper:
    """Small Language Model client for intent classification."""
    
    def __init__(self):
        self.slm = AsyncOpenAI(
            base_url=f"{SLM_URL}/v1",
            api_key='llama.cpp',  # required, but unused
        )
        self.slm_model_name = DEFAULT_SLM_MODEL
        self.messages = [
            {
                "role": "system",
                "content": INTENT_CLASSIFIER_PROMPT
            }
        ]
    
    async def classify_intent(self, user_q: str) -> str:
        """Classify user query as 'tool' or 'chat'.
        
        Args:
            user_q: User's input query
            
        Returns:
            'tool' if tool use is needed, 'chat' otherwise
        """
        self.messages.append({"role": "user", "content": f"QUERY: {user_q}"})
        response = await self.slm.chat.completions.create(
            model=self.slm_model_name,
            messages=self.messages
        )
        logger.info(f"SLM Intent Classification: {response.choices[0].message.content}")
        return response.choices[0].message.content


class MCPClientWrapper:
    """Main MCP client wrapper handling LLM and MCP server communication."""
    
    def __init__(self):
        self.model_name = DEFAULT_SLM_MODEL
        self.mcp_client = Client(f"{MCP_SERVER_URL}/mcp")
        self.llm = AsyncOpenAI(
            base_url=f"{LLAMACPP_LLM_URL}/v1",
            api_key='llama.cpp',  # required, but unused
        )
        self.slm_client = SLMClientWrapper()
        self.tools = tool_definition_list
        self.claude_messages = []
        self.result_messages = []
        self.sys_prompt = {
            "role": "system", 
            "content": SYSTEM_PROMPT
        }
        self.claude_messages.append(self.sys_prompt)
        
    async def check_connection(self) -> None:
        """Check MCP server connection."""
        async with self.mcp_client:
            await self.mcp_client.ping()
            logger.info("MCP Server is reachable")
    
    async def _connect(self) -> None:
        """Connect to MCP server and list available tools."""
        async with self.mcp_client:
            self.mcp_client.session.__aenter__
            logger.info(f"Client connected: {self.mcp_client.is_connected()}")

            # Make MCP calls within the context
            tools = await self.mcp_client.list_tools()
            logger.info(f"Connected to MCP server. Available tools: {', '.join([t.name for t in tools])}")

    def _get_file_type(self, upload_media) -> Tuple[str, Any]:
        """Determine file type from uploaded media.
        
        Args:
            upload_media: Uploaded file object
            
        Returns:
            Tuple of (file_type, data) - file_type is 'image', 'audio', or None
        """
        image_data = None
        audio_data = None
        file_type = None
        
        if upload_media:
            file_path = upload_media.name if hasattr(upload_media, 'name') else upload_media
        
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Process image
                image_data = Image.open(upload_media)
                file_type = "image"
            elif file_path.lower().endswith(('.wav', '.mp3')):
                try:
                    # Try to read and save as wav
                    wav, sr = sf.read(upload_media, dtype="float32", always_2d=False)
                except:
                    # If reading fails, try to read from bytes and convert
                    audio_bytes = upload_media.read()
                    wav, sr = read_wav_from_bytes(audio_bytes)
                
                audio_data = (wav, sr) if (wav, sr) else (None, None)
                file_type = "audio"
        
        return file_type, image_data, audio_data

    async def process_message(
        self, 
        text_query: str, 
        history: List[Union[Dict[str, Any], ChatMessage]], 
        upload_media=None
    ):
        """Process user message and stream response.
        
        Args:
            text_query: User's text input
            history: Chat history
            upload_media: Optional uploaded file (image or audio)
            
        Yields:
            Updated chat history, textbox value, image data, audio data
        """
        # Determine file type and process
        file_type, image_data, audio_data = self._get_file_type(upload_media)
        
        # Async generator to stream partial responses
        async for partial_messages, partial_image_data, partial_audio_data in self._process_query(
            text_query, history, image_data, audio_data
        ):
            yield history + partial_messages, "", partial_image_data, partial_audio_data

    async def _get_model_response_tool(
        self, 
        message: List[Dict], 
        history: List[Union[Dict[str, Any], ChatMessage]]
    ) -> Any:
        """Get LLM response with tool definitions.
        
        Args:
            message: Message history
            history: Chat history
            
        Returns:
            LLM response with tool calls if any
        """
        response = await self.llm.chat.completions.create(
            model=self.model_name,
            messages=message,
            tools=self.tools,
            tool_choice='auto'
        )
        return response
    
    async def _process_query(
        self, 
        text_query: str, 
        history: List[Union[Dict[str, Any], ChatMessage]], 
        img: Image.Image = None, 
        audio_bytes: Tuple[List, int] = None
    ):
        """Internal query processing with tool handling.
        
        Args:
            text_query: User's text input
            history: Chat history
            img: Optional PIL Image
            audio_bytes: Optional tuple of (audio array, sample rate)
            
        Yields:
            Partial messages, image data, audio data
        """
        self.result_messages = []
        
        # Build conversation history for LLM
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role", "assistant"), msg.get("content")
            
            if isinstance(content, list):
                content = content[0].get("text", "")
            
            if role in ["user", "assistant", "system"]:
                self.claude_messages.append({"role": role, "content": content})
        
        # Intent classification using SLM
        start_pre = time.perf_counter()
        pre_context_classifier = await self.slm_client.classify_intent(user_q=text_query)
        end_pre = time.perf_counter() - start_pre
        logger.info(f"[PRE_CLASSIFIER] TIME: {end_pre:.2f} {pre_context_classifier}\n")
        
        output_image_byte = None
        output_audio_data = None
        
        # Handle regular text response (chat)
        if pre_context_classifier == "chat":
            logger.info(f">>>> NO CONTEXT TRIGGERED\n\n")
            
            final_response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=self.claude_messages,
                stream=True
            )
            
            self.claude_messages = [self.sys_prompt]
            
            partial_content = ""
            async for chunk in final_response:
                if chunk.choices[0].delta.content is not None:
                    partial_content += chunk.choices[0].delta.content
                    yield [{"role": "assistant", "content": partial_content}], output_image_byte, output_audio_data
            
        elif pre_context_classifier == "tool":
            # Tool use needed
            response = await self._get_model_response_tool(self.claude_messages, history=history)
            choice = response.choices[0]
            logger.info(f"[FIRST CHOICE] - {choice}")
            first_response = choice.message
            
            logger.info(f">>>> TOOL TRIGGERED\n\n")
                    
            for tool_call in first_response.tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                logger.info(f"[TOOL] - {tool_id} - {tool_name}")
                
                # Parse tool arguments
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"raw_args": tool_call.function.arguments}
                
                # Add tool use messages to history
                self._add_tool_messages(tool_name, tool_id, tool_args)
                
                result = await self._call_tool(tool_name, tool_args, img, audio_bytes)
                
                # Add tool response to LLM messages
                await self._add_tool_response(tool_name, tool_id, tool_args, result)
            
            # Get final response after tool use
            final_response = await self.llm.chat.completions.create(
                model=self.model_name,
                messages=self.claude_messages,
                stream=True
            )
            
            self.claude_messages = [self.sys_prompt]
            
            partial_content = ""
            async for chunk in final_response:
                if chunk.choices[0].delta.content is not None:
                    partial_content += chunk.choices[0].delta.content
                    yield [{"role": "assistant", "content": partial_content}], output_image_byte, output_audio_data

    def _add_tool_messages(self, tool_name: str, tool_id: str, tool_args: Dict) -> None:
        """Add tool call messages to result history.
        
        Args:
            tool_name: Name of the tool
            tool_id: Tool call ID
            tool_args: Tool arguments
        """
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
        
        self.result_messages.append({
            "role": "assistant",
            "content": "```json\n" + json.dumps(tool_args, indent=2, ensure_ascii=True) + "\n```",
            "metadata": {
                "parent_id": f"tool_call_{tool_name}",
                "id": f"params_{tool_name}",
                "title": "Tool Parameters"
            }
        })
        
        logger.info(f"[RESULT MES] AFTER TOOL: {self.result_messages}")

    async def _call_tool(
        self, 
        tool_name: str, 
        tool_args: Dict, 
        img: Image.Image, 
        audio_bytes: Tuple
    ) -> Any:
        """Call the appropriate MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Tool arguments
            img: Image data
            audio_bytes: Audio data
            
        Returns:
            Tool result
        """
        result = "Fail to get the server response"
        
        logger.info(f"TOOL ARGS: {tool_args}")
        
        if tool_name == "generate_image":
            result = await call_image_generation_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
            if result and isinstance(result, CallToolResult) and hasattr(result, "data"):
                image_bytes = base64.b64decode(result.data)
                image = Image.open(io.BytesIO(image_bytes))
                return image
                
        elif tool_name == "describe_image":
            result = await call_image_describe_tool(
                file_byte=img,
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
        elif tool_name == "transcribe_audio":
            result = await call_transcribe_audio_tool(
                audio_data=audio_bytes,
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_forecast":
            result = await call_get_forecast_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_alerts":
            result = await call_get_alerts_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_multiply":
            result = await call_get_multiply_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.result_messages,
                tool_name=tool_name
            )
            
        logger.info(f"[RESULT MES] RESULT TOOL CALL MESSAGE: {self.result_messages}")
        return result

    async def _add_tool_response(
        self, 
        tool_name: str, 
        tool_id: str, 
        tool_args: Dict, 
        result: Any
    ) -> None:
        """Add tool response to LLM message history.
        
        Args:
            tool_name: Name of the tool
            tool_id: Tool call ID
            tool_args: Tool arguments
            result: Tool result
        """
        # Add tool call to messages
        self.claude_messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    }
                }
            ]
        })
        
        # Add tool response
        if tool_name == "generate_image":
            tool_response = await add_image_tool_response(
                result=tool_args.get("prompt", ""),
                tool_id=tool_id,
                tool_name=tool_name
            )
        else:
            tool_response = await add_tool_response(
                tool_name=tool_name,
                result=result,
                tool_id=tool_id
            )
                
        logger.info(f"[TOOL RESPONSE] {tool_response}")
        self.claude_messages.append(tool_response)


# Import base64 and io for image processing
import base64
import io

