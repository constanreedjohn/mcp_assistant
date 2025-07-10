from typing import List, Dict, Any, Union

def get_tool_definition():
    return [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image using SanaSprint model with the output of three parameters: 'prompt', 'width', 'height'",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt describing the image to generate"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width (default: 512)",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height (default: 512)",
                        "default": 512
                    }
                },
                "required": ["prompt"]
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Describe the image with the appropriate prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt about the detail requirement for the image description."
                    },
                },
                "required": ["prompt"]
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alerts",
            "description": "Get weather alerts for a US state from an API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter US state code (e.g. CA, NY)"
                    },
                },
                "required": ["state"]
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get weather forecast for a location from an API",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "string",
                        "description": "Latitude of the location"
                    },
                    "longtitude": {
                        "type": "string",
                        "description": "longtitude of the location"
                    },
                },
                "required": ["latitude", "longtitude"],
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_multiply",
            "description": "Calculate multiplication between a and b",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_number": {
                        "type": "integer",
                        "description": "first number"
                    },
                    "second_number": {
                        "type": "integer",
                        "description": "second number"
                    },
                },
                "required": ["first_number", "second_number"],
            },
            "strict": True,
        },
    },
]
    
tool_definition_list = get_tool_definition()