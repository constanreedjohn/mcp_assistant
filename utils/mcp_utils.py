import json
import base64
import traceback
from fastmcp import Client
from typing import List, Dict, Optional
from io import BytesIO
from PIL import Image
from fastmcp.client.client import CallToolResult


async def call_image_generation_tool(
    mcp_client: Client, 
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict] | CallToolResult | str | Dict:
    
    try:
        async with mcp_client:
            result: CallToolResult = await mcp_client.call_tool(
                name="generate_image", 
                arguments=dict(
                    prompt=tool_args.get("prompt", ""),
                    width=tool_args.get("width", 512),
                    height=tool_args.get("height", 512)
                )
            )
        
        # Update the status of the tool call
        if result_messages and "metadata" in result_messages[-2]:
            result_messages[-2]["metadata"]["status"] = "done"
        
        # Add a header for the tool results
        result_messages.append({
            "role": "assistant",
            "content": "Here are the results from the tool:",
            "metadata": {
                "title": f"Tool Result for {tool_name}",
                "status": "done",
                "id": f"result_{tool_name}"
            }
        })
    except TimeoutError as e:
        print("Request timed out")
        result = [{"text": "Request timed out"}]
        result_messages.append({
            "role": "assistant",
            "content": "```\nRequest timed out\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        raise TimeoutError(e)
    except Exception as e_else:
        result = [{"text": "Fail to get the server response"}]
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        raise Exception(e_else)
    
    print(f"[TOOL CALL] Generate_image: Result - {result} - {type(result)}")
    result_content = result.data
    byte_data = base64.b64decode(result_content)
                    
    try:
        print(f"[TOOL CALL] result JSON: {result_content}")
        if isinstance(byte_data, bytes):
            result_messages.append({
                "role": "assistant",
                "content": "Here are the results from the tool:",
                "metadata": {
                    "parent_id": f"result_{tool_name}",
                    "id": f"image_{tool_name}",
                    "status": "done",
                    "title": "Generated Image"
                }
            })
        else:
            result_messages.append({
                "role": "assistant",
                "content": "Failed to get the generated image.",
                "metadata": {
                    "parent_id": f"result_{tool_name}",
                    "id": f"raw_result_{tool_name}",
                    "status": "done",
                    "title": "Raw Output"
                }
            })
    except Exception as err:
        result_messages.append({
            "role": "assistant",
            "content": "Failed to get the generated image.",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "title": "Raw Output",
                "status": "done",
            }
        })
        raise Exception(err)
    return result


async def call_image_describe_tool(
    mcp_client: Client,
    file_byte: Image.Image,
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict] | CallToolResult | str | Dict:
    
    try:
        file_byte.save("../input.jpg")
        async with mcp_client:
            # Convert bytes to base64 string for sending
            # b64_image = base64.b64encode(file_bytes).decode('utf-8')
            result: CallToolResult = await mcp_client.call_tool(
                name="describe_image",
                arguments=dict(
                    prompt=tool_args.get("prompt", "")
                )
            )
        
        # Update the status of the tool call
        if result_messages and "metadata" in result_messages[-2]:
            result_messages[-2]["metadata"]["status"] = "done"
        
        # Add a header for the tool results
        result_messages.append({
            "role": "assistant",
            "content": "Here are the results from the tool:",
            "metadata": {
                "title": f"Tool Result for {tool_name}",
                "status": "done",
                "id": f"result_{tool_name}"
            }
        })
    except Exception as e:
        print(traceback.format_exc(), e)
        result = "Fail to get the server response."
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        raise Exception(e)
        
    print(f"[TOOL CALL] Image_analyze: Result - {result} - {type(result)}")
    # Process the result content - assuming result is a string
    result_content: dict = result.structured_content
    content_message = f"The description of the image is {result_content['message']}"
    
    result_messages.append({
        "role": "assistant",
        "content": "```\n" + content_message + "\n```",
        "metadata": {
            "parent_id": f"result_{tool_name}",
            "id": f"raw_result_{tool_name}",
            "title": "Raw Output",
            "status": "done",
        }
    })
        
    return content_message

async def call_get_forecast_tool(
    mcp_client: Client,
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict] | CallToolResult | str | Dict:
    try:
        async with mcp_client:
            result: str = await mcp_client.call_tool(
                name="get_forecast", 
                arguments=dict(
                    latitude=tool_args.get("latitude"),
                    longtitude=tool_args.get("longtitude")
                )
            )
        print(f"[TOOL CALL] Get_forecast: Result - {result}")
        
        # Update the status of the tool call
        if result_messages and "metadata" in result_messages[-2]:
            result_messages[-2]["metadata"]["status"] = "done"
        
        # Add a header for the tool results
        result_messages.append({
            "role": "assistant",
            "content": "Here are the results from the tool:",
            "metadata": {
                "title": f"Tool Result for {tool_name}",
                "status": "done",
                "id": f"result_{tool_name}"
            }
        })
        
        # Process the result content - assuming result is a string
        result_content = result
        
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result_content + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "title": "Raw Output",
                "status": "done",
            }
        })
        
        print(f"[RESULT MES] RESULT TOOL CALL MESSAGE: {result_messages}")
    except:
        result = "Fail to get the server response"
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        
    return result

async def call_get_alerts_tool(
    mcp_client: Client,
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict] | CallToolResult | str | Dict:
    try:
        async with mcp_client:
            result: str = await mcp_client.call_tool(
                name="get_alerts",
                arguments=dict(
                    state=tool_args.get("state")
                )
            )
            
        print(f"[TOOL CALL] Get_alerts: Result - {result}")
        
        # Update the status of the tool call
        if result_messages and "metadata" in result_messages[-2]:
            result_messages[-2]["metadata"]["status"] = "done"
        
        # Add a header for the tool results
        result_messages.append({
            "role": "assistant",
            "content": "Here are the results from the tool:",
            "metadata": {
                "title": f"Tool Result for {tool_name}",
                "status": "done",
                "id": f"result_{tool_name}"
            }
        })
        
        # Process the result content - assuming result is a string
        result_content = result
        
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result_content + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        print(f"[RESULT MES] RESULT TOOL CALL MESSAGE: {result_messages}")
    
    except:
        result = "Fail to get the server response"
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        
    return result

async def call_get_multiply_tool(
    mcp_client: Client,
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict] | CallToolResult | str | Dict:
    try:
        async with mcp_client:
            result: CallToolResult = await mcp_client.call_tool(
                name="get_multiply", 
                arguments=dict(
                    first_number=tool_args.get("first_number"),
                    second_number=tool_args.get("second_number")
                )
            )
            
        print(f"[TOOL CALL] Get_Multiply: Result - {result}")
        
        # Update the status of the tool call
        if result_messages and "metadata" in result_messages[-2]:
            result_messages[-2]["metadata"]["status"] = "done"
        
        result_content = result.structured_content
        
        # Add a header for the tool results
        result_messages.append({
            "role": "assistant",
            "content": "Here are the results from the tool:",
            "metadata": {
                "title": f"Tool Result for {tool_name}",
                "status": "done",
                "id": f"result_{tool_name}"
            }
        })
        
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result_content["message"] + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
    except:
        result = "Fail to get the server response"
        result_messages.append({
            "role": "assistant",
            "content": "```\n" + result + "\n```",
            "metadata": {
                "parent_id": f"result_{tool_name}",
                "id": f"raw_result_{tool_name}",
                "status": "done",
                "title": "Raw Output"
            }
        })
        
    return result

def add_tool_response(
    result: List[Dict] | Dict | str | CallToolResult,
    tool_id: str,
    tool_name: str
) -> Optional[Dict] | Optional[str]:
    
    tool_response = {
        "role": "tool",
        "tool_name": tool_name,
        "tool_call_id": tool_id,
        "content": ""
    }
    if isinstance(result, CallToolResult):
        tool_response["content"] = result.structured_content["message"]
        return tool_response
    if isinstance(result, dict):
        tool_response["content"] = result["message"]
        return tool_response
    if isinstance(result, str):
        tool_response["content"] = result
        return tool_response
    else:
        try:
            result_json = json.loads(result)    
            tool_response["content"] = result_json["message"]
        except:
            tool_response["content"] = result    
        return tool_response

def add_image_tool_response(
    result: str,
    tool_id: str,
    tool_name: str
) -> Optional[Dict] | Optional[str]:
    try:
        tool_response = {
            "role": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_id,
            "content": f"Image generated successfully with prompt {result}"
        }
    except:
        tool_response = {
            "role": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_id,
            "content": "Image generated Failed."
        }
    
    print(f"[TOOL RESPONSE] {tool_response}")
    return tool_response