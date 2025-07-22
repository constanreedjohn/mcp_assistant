import requests
import traceback
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
        if response.status_code != 200:
            return Exception(f"Fail to get response")
        else:
            return response.json()
    except Exception as e:
        logger.info(f"Fail to get response - {e}")
        logger.info(traceback.format_exc())
        return e

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