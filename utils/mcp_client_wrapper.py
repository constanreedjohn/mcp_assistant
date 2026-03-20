"""
MCP Client Wrapper for the MCP Assistant.
Handles communication with the MCP server and LLM integration.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
import time
import traceback
from typing import List, Dict, Any, Union, Tuple
from PIL import Image

import soundfile as sf
from fastmcp import Client
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from gradio.components.chatbot import ChatMessage

from .tools import tool_definition_list
from .logging_utils import logger
from .audio_utils import read_wav_from_bytes, encode_image
from .constants import (
    SYSTEM_PROMPT, 
    TOOL_MONITOR_SYS_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    DEFAULT_LLM_MODEL,
    DEFAULT_SLM_MODEL,
    DEFAULT_INPUT_IMAGE,
)
from config import (
    MCP_SERVER_URL, 
    LLAMACPP_LLM_URL, 
    SLM_URL, 
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    DOCUMENT_STORAGE_DIR
)
from .mcp_utils import (
    call_get_forecast_tool,
    call_get_alerts_tool,
    call_get_multiply_tool,
    call_transcribe_audio_tool,
    add_tool_response,
    add_image_tool_response,
    add_document_tool_response,
)
from pathlib import Path
from .rag_utils import RAGPipeline


class SLMClientWrapper:
    """Small Language Model client for intent classification."""
    
    def __init__(self):
        self.slm = AsyncOpenAI(
            base_url=f"{SLM_URL}/v1",
            api_key='llama.cpp',  # required, but unused
        )
        self.slm_model_name = DEFAULT_SLM_MODEL # DEFAULT_SLM_MODEL
    
    async def clean_context(self, user_q):
        self.messages = [
            {
                "role": "system",
                "content": INTENT_CLASSIFIER_PROMPT.format(query_message=user_q)
            }
        ]
        
    async def classify_intent(self, user_q: str) -> str:
        """Classify user query as 'tool' or 'chat'.
        
        Args:
            user_q: User's input query
            
        Returns:
            'tool' if tool use is needed, 'chat' otherwise
        """
        self.messages = [
            {
                "role": "system",
                "content": INTENT_CLASSIFIER_PROMPT.format(query_message=user_q)
            }
        ]
        self.messages.append({"role": "user", "content": f"Please give me the response."})
        
        response = await self.slm.chat.completions.create(
            model=self.slm_model_name,
            messages=self.messages
        )
        logger.info(f"SLM Intent Classification: {response.choices[0].message.content}")
        
        await self.clean_context(user_q)
        
        return response.choices[0].message.content


class MCPClientWrapper:
    """Main MCP client wrapper handling LLM and MCP server communication."""
    
    def __init__(self):
        # LLM Init
        self.model_name = DEFAULT_LLM_MODEL
        self.mcp_client = Client(f"{MCP_SERVER_URL}/mcp")
        self.llm = AsyncOpenAI(
            base_url=f"{LLAMACPP_LLM_URL}/v1",
            api_key='llama.cpp',  # required, but unused
        )
        self.slm_client = SLMClientWrapper()
        
        # RAG Init
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.rag_pipeline = RAGPipeline(
            qdrant_client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chunk_tokens=CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            storage_dir=DOCUMENT_STORAGE_DIR
        )
        
        # Main LLM Chatbot Inint
        self.tools = tool_definition_list
        self.claude_messages = []
        self.tool_monitor_messages = []
        self.tool_sys_prompt = {
            "role": "system", 
            "content": TOOL_MONITOR_SYS_PROMPT
        }
        self.sys_prompt = {
            "role": "system", 
            "content": SYSTEM_PROMPT
        }
        self.claude_messages.append(self.sys_prompt)
        self.tool_monitor_messages.append(self.tool_sys_prompt)
        
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

    async def _stream_final_response(self, client: AsyncOpenAI, prompt_series: List):
        # Stream final response
        final_response = await client.chat.completions.create(
            model=self.model_name,
            messages=prompt_series,
            stream=True
        )
        
        self.claude_messages = [self.sys_prompt]
        
        partial_content = ""
        async for chunk in final_response:
            if chunk.choices[0].delta.content is not None:
                partial_content += chunk.choices[0].delta.content
                yield [{"role": "assistant", "content": partial_content}]
    
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
    
    def _get_file_type(self, upload_media) -> Tuple[str, Any]:
        """Determine file type from uploaded media.
        
        Args:
            upload_media: Uploaded file object
            
        Returns:
            Tuple of (file_type, data) - file_type is 'image', 'audio', 'document', or None
        """
        image_data = None
        audio_data = None
        file_type = None
        
        if upload_media:
            file_path = upload_media.name if hasattr(upload_media, 'name') else upload_media
        
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Process image
                image_data = Image.open(upload_media)
                image_data.save(DEFAULT_INPUT_IMAGE)
                file_type = "image"
            elif file_path.lower().endswith(('.wav', '.mp3')):
                try:
                    # Try to read and save as wav
                    wav, sr = sf.read(upload_media, dtype="float32", always_2d=False)
                except:
                    # If reading fails, try to read from bytes and convert
                    with open(file_path, 'rb') as fin:
                        audio_bytes = fin.read()
                    
                    wav, sr = read_wav_from_bytes(audio_bytes)
                
                audio_data = (wav, sr) if (wav, sr) else (None, None)
                file_type = "audio"
        
        return file_type, image_data, audio_data

    def _process_document_file(
        self,
        files
    ):
        logger.info(f"[INGEST DOCUMENT] INGESTING DOCUMENT(S)...")
        
        # Normalize to list
        if isinstance(files, str):
            files = [files]
        elif files is None:
            files = []
        
        processed = 0
        for file_path in files:
            if not file_path or not file_path.lower().endswith('.pdf'):
                logger.info(f"[INGEST DOCUMENT] Skipping non-PDF: {file_path}")
                continue
                
            try:
                filename = Path(file_path).name
                logger.info(f"[INGEST DOCUMENT] Processing {filename}...")
                
                with open(file_path, 'rb') as fin:
                    file_bytes = fin.read()
                
                logger.info(f"[INGEST DOCUMENT] Read {len(file_bytes)} bytes from {filename}")
                
                # Unique ID per file
                document_id = str(uuid.uuid4())
                
                result = self.rag_pipeline.ingest_document(
                    content=file_bytes,
                    document_id=document_id,
                    filename=filename
                )
                logger.info(f"[INGEST DOCUMENT] Ingested {filename} (ID: {document_id}, chunks: {result['num_chunks']})")
                processed += 1
                
            except Exception as e:
                logger.error(f"[DOCUMENT][INGESTION] Failed {file_path}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"[INGEST DOCUMENT] Completed processing {processed}/{len(files)} PDF(s)")

                
    async def _process_message(
        self, 
        text_query: str, 
        history: List[Union[Dict[str, Any], ChatMessage]], 
        upload_media=None,
        rag_enabled: bool = True,
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
        async for partial_messages in self._process_main_query(
            text_query, history, image_data, audio_data, rag_enabled
        ):
            yield history + partial_messages, ""
    
    async def _process_main_query(
        self, 
        text_query: str, 
        history: List[Union[Dict[str, Any], ChatMessage]], 
        img: Image.Image = None, 
        audio_bytes: Tuple[List, int] = None,
        rag_enabled: bool = True,
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
        
        self.tool_monitor_messages = [self.tool_sys_prompt]
                
        # ============================================================================
        # Build conversation history
        # ============================================================================
        
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role", "assistant"), msg.get("content")
            
            if isinstance(content, list):
                content = content[0].get("text", "")
            
            if role in ["user", "assistant", "system"]:
                self.claude_messages.append({"role": role, "content": content})
        
        # ============================================================================
        # RAG mode
        # ============================================================================
        if rag_enabled:
            logger.info(f"[RAG] TRIGGERED RAG MODE.")
            try:
                if not self.rag_pipeline:
                    logger.info(f"[RAG] Error - RAG pipeline not initialized")
                
                # Retrieve documents
                result = await self.rag_pipeline.retrieve(
                    query=text_query,
                    limit=10,
                    document_id=None,
                    validate=False
                )
                
                # Post process retrieved chunks into main LLM
                rag_response = await add_document_tool_response(
                    result=result,
                    tool_id=None,
                    tool_name=None
                )
                
                # Add RAG prompt into the main LLM prompts.
                rag_prompt = await self.rag_pipeline.retrieval_engine.get_response_prompt(
                    query=result['query'],
                    retrieved_document=rag_response["content"]
                )
                self.claude_messages.append(rag_prompt)
                logger.info(f"[RAG] CLAUDE MESSAGE WITH RAG: {self.claude_messages}")
                
                # Stream response - consume async gen
                async for chunk in self._stream_final_response(client=self.llm, prompt_series=self.claude_messages):
                    yield chunk
            
            except Exception as e:
                logger.info(traceback.format_exc())
                logger.info(f"[RAG] Error: {str(e)}")
            
        else:
            # ============================================================================
            # Intent classification using SLM
            # ============================================================================
            
            start_pre = time.perf_counter()
            pre_context_classifier = await self.slm_client.classify_intent(user_q=text_query)
            end_pre = time.perf_counter() - start_pre
            logger.info(f"[PRE_CLASSIFIER] TIME: {end_pre:.2f} {pre_context_classifier}\n")
            
            output_image_byte = None
            output_audio_data = None
            
            # ============================================================================
            # Handle regular text response (chat)
            # ============================================================================

            if pre_context_classifier == "chat":
                logger.info(f">>>> NO CONTEXT TRIGGERED\n\n")
                
                # Handle image description if available
                if img:
                    base64_image = await encode_image(DEFAULT_INPUT_IMAGE)
                    
                    self.claude_messages += [
                        {
                            "role": "user",
                            "content": [
                                { "type": "text", "text": text_query },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    }
                                },
                            ],
                        }
                    ]
                
                # Stream response - consume async gen
                async for chunk in self._stream_final_response(client=self.slm_client.slm, prompt_series=self.claude_messages):
                    yield chunk
                
            # ============================================================================
            # MCP Tool Calling
            # ============================================================================
            
            elif pre_context_classifier == "tool":
                # Tool function router/decision
                response = await self._get_model_response_tool(self.claude_messages, history=history)
                choice = response.choices[0]
                logger.info(f"[FIRST CHOICE] - {choice}")
                first_response = choice.message
                
                logger.info(f">>>> TOOL TRIGGERED\n\n")

                # There maybe more than one tool used - handle tools sequentially.
                if first_response.tool_calls:
                    for tool_call in first_response.tool_calls:
                        tool_id = tool_call.id
                        tool_name = tool_call.function.name
                        logger.info(f"[TOOL] - {tool_id} - {tool_name}")
                        
                        # Parse generated tool arguments from the LLM
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {"raw_args": tool_call.function.arguments}
                        
                        # Add tool use messages to main prompt
                        await self._add_tool_messages(tool_name, tool_id, tool_args)
                        
                        # # Stream tool monitor response
                        # async for chunk in self._stream_final_response(client=self.slm_client.slm, prompt_series=self.tool_monitor_messages):
                        #     yield chunk
                        
                        # Call MCP tool
                        result = await self._call_tool(tool_name, tool_args, audio_bytes)
                        
                        # Stream tool monitor response
                        async for chunk in self._stream_final_response(client=self.slm_client.slm, prompt_series=self.tool_monitor_messages):
                            yield chunk
                        
                        # Add tool response to LLM main prompt
                        await self._add_tool_response(tool_name, tool_id, tool_args, result)
                        
                        # Get final response after tool use
                        async for chunk in self._stream_final_response(client=self.llm, prompt_series=self.claude_messages):
                            yield chunk
                else:
                    # Incase no tools selected - Handle normal text in the LLM reponse.
                    self.claude_messages.append({"role": "assistant", "content": first_response.content})
            
                    # Get final response if no tools used.
                    async for chunk in self._stream_final_response(client=self.llm, prompt_series=self.claude_messages):
                        yield chunk

    async def _add_tool_messages(self, tool_name: str, tool_id: str, tool_args: Dict) -> None:
        """Add tool call messages to result history.
        
        Args:
            tool_name: Name of the tool
            tool_id: Tool call ID
            tool_args: Tool arguments
        """
        self.tool_monitor_messages.append({
            "role": "assistant",
            "content": f"I'll use the {tool_name} tool to help answer your question.",
            "metadata": {
                "title": f"Using tool: {tool_name}",
                "log": f"Parameters: {json.dumps(tool_args, ensure_ascii=True)}",
                "status": "pending",
                "id": f"tool_call_{tool_name}"
            }
        })
        
        self.tool_monitor_messages.append({
            "role": "assistant",
            "content": "```json\n" + json.dumps(tool_args, indent=2, ensure_ascii=True) + "\n```",
            "metadata": {
                "parent_id": f"tool_call_{tool_name}",
                "id": f"params_{tool_name}",
                "title": "Tool Parameters"
            }
        })
        
        self.tool_monitor_messages.append({
            "role": "user",
            "content": "Give me the response of the tool monitoring including tool_name and tool_parameters",
            "metadata": {
                "parent_id": f"tool_call_{tool_name}",
                "id": f"params_{tool_name}",
                "title": "Tool Parameters"
            }
        })
        
        
        logger.info(f"[RESULT MES] AFTER TOOL: {self.tool_monitor_messages}")

    async def _call_tool(
        self, 
        tool_name: str, 
        tool_args: Dict,
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
        
        if tool_name == "transcribe_audio":
            result = await call_transcribe_audio_tool(
                audio_data=audio_bytes,
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.tool_monitor_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_forecast":
            result = await call_get_forecast_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.tool_monitor_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_alerts":
            result = await call_get_alerts_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.tool_monitor_messages,
                tool_name=tool_name
            )
        elif tool_name == "get_multiply":
            result = await call_get_multiply_tool(
                mcp_client=self.mcp_client,
                tool_args=tool_args,
                result_messages=self.tool_monitor_messages,
                tool_name=tool_name
            )
            
        self.tool_monitor_messages.append({
            "role": "user",
            "content": "Response the tool status and the tool response for me.",
        })
            
        logger.info(f"[RESULT MES] RESULT TOOL CALL MESSAGE: {self.tool_monitor_messages}")
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

