from typing import List, Dict, Any, Union

def get_tool_definition():
    return [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "Retrieve relevant context from the documents to provide information for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A text prompt of the user request."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The number of the retrieved chunks"
                    },
                    "validate": {
                        "type": "boolean",
                        "description": "Whether to validate the context of the retrieved chunks."
                    },
                },
                "required": ["query"]
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe from a given audio file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt about the detail requirement for the audio file."
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