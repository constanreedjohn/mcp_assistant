import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import requests
from fastmcp import FastMCP, Context, Image

from utils.logging_utils import logger
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from utils.utils import make_nws_request, format_alert, NWS_API_BASE

mcp = FastMCP(name="MainMcpServer", timeout_keep_alive=1000000000)

custom_middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"]),
]

@mcp.tool(
    annotations={
        "title": "Calculate multiplication"
    }
)
async def get_multiply(first_number: int, second_number: int, ctx: Context) -> dict:
    """Multiply two intergers together and return a JSON object for status.
    """
    logger.info(f"[SERVER][GET MULTIPLY] Triggered")
    
    c = first_number * second_number
    
    logger.info(f"[SERVER][GET MULTIPLY] Done - Result {c}")
    
    return {
        "status": "ok",
        "message": f"the mutiplication is {c}"
    }
    
@mcp.tool(
    annotations={
        "title": "Get weather alerts for US state from external API"
    }
)
async def get_alerts(state: str, ctx: Context) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
            
    logger.info(f"[SERVER][GET_ALERTS] Triggered")
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    
    logger.info(f"[SERVER][GET_ALERTS] Done")
    return "\n---\n".join(alerts)

@mcp.tool(
    annotations={
        "title": "Get weather forecast for a location from external API"
    }
)
async def get_forecast(latitude: float, longtitude: float, ctx: Context) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longtitude: longtitude of the location
    """
    logger.info(f"[SERVER][GET_FORECAST] Triggered")
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longtitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    logger.info(f"[SERVER][GET_FORECAST] Done")
    return "\n---\n".join(forecasts)

@mcp.tool(
    annotations={
        "title": "Generate image from a locally deploy Image Diffusion/Denoising model."
    }
)
async def generate_image(prompt: str, ctx: Context, width: int = 512, height: int = 512) -> Image | None | dict:
    """Call request to image generator API
    
    Args:
        prompt: Text prompt describing the image to generate
        width: Image width (default: 512)
        height: Image height (default: 512)
    """
    logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Triggered")
    try:
        params = {
            "prompt": prompt,
            "width": width,
            "height": height
        }
        response = requests.get(f"http://localhost:3001/image/generate", params=params)
        # First check if the request was successful (HTTP 200)
        if response.status_code == 200:
            try:
                data = response.json()  # Use response.json() instead of json.loads(response.content)
                
                if data["status"] == "success":                    
                    logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Got reponse {data['message']}")
                    logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Done")
                    image = Image(data=base64.b64decode(data["image_bytes"]), format="png")
                    return image.to_image_content()
                else:
                    logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Failed: {data}")
                    return data
                
            except ValueError:
                logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Failed: Invalid JSON response")
                return {
                    "status": "error",
                    "message": "Invalid response format from image generation service"
                }
        else:
            logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Failed: HTTP {response.status_code}")
            return {
                "status": "failed",
                "message": f"Image generation service returned HTTP {response.status_code}"
            }
        
    except Exception as e:
        logger.info(f"[SERVER][GEN_LOCAL_IMAGE] Done")
        return {
            "status": "failed",
            "message": "Failed to generate image"
        }

http_app = mcp.http_app(middleware=custom_middleware)

async def main():
    # Use run_async() in async contexts
    await mcp.run_http_async(
        transport="streamable-http",
        host="127.0.0.1",
        port=5001
    )
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())