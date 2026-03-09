"""
Weather API utilities for the MCP Assistant.
Handles NWS (National Weather Service) API requests.
"""
import requests
import traceback
from typing import Any, Dict

from .logging_utils import logger
from config import NWS_API_BASE, USER_AGENT


async def make_nws_request(url: str) -> Dict[str, Any]:
    """Make a request to the NWS API with proper error handling.
    
    Args:
        url: Full URL to the NWS API endpoint
        
    Returns:
        JSON response as dict, or Exception on failure
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30.0)
        logger.info(f"+++ GOT RESPONSE: {response.status_code}")
        if response.status_code != 200:
            return Exception(f"Fail to get response: HTTP {response.status_code}")
        return response.json()
    except Exception as e:
        logger.info(f"Fail to get response - {e}")
        logger.info(traceback.format_exc())
        return e


def format_alert(feature: Dict) -> str:
    """Format an alert feature into a readable string.
    
    Args:
        feature: Alert feature from NWS API
        
    Returns:
        Formatted alert string
    """
    props: Dict = feature.get("properties", {})
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


async def get_weather_alerts(state: str) -> str:
    """Get weather alerts for a US state.
    
    Args:
        state: Two-letter US state code (e.g. CA, NY)
        
    Returns:
        Formatted alert string or error message
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


async def get_weather_forecast(latitude: str, longitude: str) -> str:
    """Get weather forecast for a location.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        
    Returns:
        Formatted forecast string or error message
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    try:
        points_data: Dict = await make_nws_request(points_url)
    except Exception as e:
        return f"Unable to fetch forecast data for this location. - {e}"
    
    # Get the forecast URL from the points response
    forecast_url = points_data.get("properties", {}).get("forecast")
    if not forecast_url:
        return f"Unable to get forecast URL for location {latitude}, {longitude}"
    
    try:
        forecast_data = await make_nws_request(forecast_url)
    except Exception as e:
        return f"Failed to get the URL forecast at {forecast_url} - {e}"
    
    try:
        # Format the periods into a readable forecast
        periods = forecast_data.get("properties", {}).get("periods", [])
        forecasts = []
        for period in periods[:5]:  # Only show next 5 periods
            forecast = f"""
    {period.get('name', 'Unknown')}:
    Temperature: {period.get('temperature', '?')}°{period.get('temperatureUnit', 'F')}
    Wind: {period.get('windSpeed', '?')} {period.get('windDirection', '')}
    Forecast: {period.get('detailedForecast', 'No forecast available')}
    """
            forecasts.append(forecast)

        return "\n---\n".join(forecasts)
    
    except Exception as e:
        return f"Failed to get the server response with location {latitude} {longitude} - {e}"

