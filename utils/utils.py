import requests
from typing import Any

from .logging_utils import logger

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

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
        logger.info("Fail to get response")
        return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props: dict = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""