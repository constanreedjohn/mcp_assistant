import json
import base64
from fastmcp import Client
from typing import List, Dict, Optional


async def call_image_generation_tool(
    mcp_client: Client, 
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str) -> List[Dict] | List:
    
    try:
        async with mcp_client:
            result = await mcp_client.call_tool(
                name="generate_image", 
                arguments=dict(
                    prompt=tool_args.get("prompt", ""),
                    width=tool_args.get("width", 512),
                    height=tool_args.get("height", 512)
                )
            )
        print(f"[TOOL CALL] Generate_image: Result - {result} - {type(result)}")
        
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
    except TimeoutError:
        print("Request timed out")
    except:
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
        
    result_content = result[0].data
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
    except:
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
    return result

async def call_get_forecast_tool(
    mcp_client: Client,
    tool_args: Dict,
    result_messages: List[Dict],
    tool_name: str
) -> List[Dict]:
    try:
        async with mcp_client:
            result = await mcp_client.call_tool(
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
        result_content = result[0].text
        
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
) -> List[Dict]:
    try:
        async with mcp_client:
            await mcp_client.ping()
            result = await mcp_client.call_tool(
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
        result_content = result[0].text
        
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
) -> List[Dict]:
    try:
        async with mcp_client:
            result = await mcp_client.call_tool(
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
        
        result_content = result[0].text
        temp = json.loads(result_content)
        
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
            "content": "```\n" + temp["message"] + "\n```",
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
    result: List[Dict] | Dict | str,
    tool_id: str
) -> Optional[Dict] | Optional[str]:
    try:
        result_json = json.loads(result)
        tool_response = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result_json["message"]
        }
    except:
        tool_response = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result
        }
    
    print(f"[TOOL RESPONSE] {tool_response}")
    return tool_response

def add_image_tool_response(
    result: str,
    tool_id: str
) -> Optional[Dict] | Optional[str]:
    try:
        tool_response = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": f"Image generated successfully with prompt {result}"
        }
    except:
        tool_response = {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": "Image generated Failed."
        }
    
    print(f"[TOOL RESPONSE] {tool_response}")
    return tool_response