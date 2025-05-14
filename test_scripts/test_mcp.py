from typing import Any
import requests
from fastmcp import Client
import asyncio

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"
url = f"{NWS_API_BASE}/alerts/active/area/TX"

async def make_nws_request(url: str) -> dict[str, Any]:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30.0)
        print(f"+++ GOT RESPONSE: {response.status_code}")
        response.raise_for_status()  # Raises an exception for 4XX/5XX responses
        return response.json()
    except Exception:
        print("Fail to get response")
        return None
    
async def test_get_multiply():
    client = Client("http://127.0.0.1:5001/mcp")
    async with client:
        await client.ping()
        rest = await client.call_tool(
            name="get_multiply",
            arguments=dict(
                first_number=3,
                second_number=4
            )
        )
        print(rest)

async def test_get_alerts():
    client = Client("http://127.0.0.1:5001/mcp")
    async with client:
        await client.ping()
        rest = await client.call_tool(
            name="get_alerts",
            arguments=dict(
                state="TX"
            )
        )
        print(rest)
        
async def test_get_forecast():
    client = Client("http://127.0.0.1:5001/mcp")
    async with client:
        await client.ping()
        rest = await client.call_tool(
            name="get_forecast",
            arguments=dict(
                latitude=31.9686,
                longtitude=99.9018
            )
        )
        print(rest)
        
async def test_generate_image():
    client = Client("http://127.0.0.1:5001/mcp")
    async with client:
        await client.ping()
        rest = await client.call_tool(
            name="generate_image",
            arguments=dict(
                prompt="A wooden table"
            )
        )
        print(rest)
    
async def main():
    # await test_get_multiply(),
    # await test_get_alerts()
    # await make_nws_request(url)
    # await test_get_forecast()
    await test_generate_image()

if __name__ == "__main__":
    asyncio.run(main())