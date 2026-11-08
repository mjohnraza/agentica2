"""JSON tool schemas for the SaafHawa agent."""

from typing import Any, Dict, List

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_aqi",
            "description": "Fetch real-time air quality index (AQI) and pollutant concentrations (PM2.5, PM10, NO2, O3) for a specified location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city or municipality (e.g., 'Karachi', 'Lahore').",
                    },
                    "lat": {
                        "type": "number",
                        "description": "Latitude coordinate of the target observation point.",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude coordinate of the target observation point.",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "coordinates": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                        },
                        "required": ["lat", "lon"],
                    },
                    "aqi": {"type": "integer", "description": "US EPA standard AQI value (0-500)."},
                    "category": {
                        "type": "string",
                        "enum": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
                    },
                    "pollutants": {
                        "type": "object",
                        "properties": {
                            "pm2_5": {"type": "number", "description": "PM2.5 concentration in µg/m³"},
                            "pm10": {"type": "number", "description": "PM10 concentration in µg/m³"},
                            "no2": {"type": "number", "description": "NO2 concentration in ppb"},
                            "o3": {"type": "number", "description": "Ozone concentration in ppb"},
                        },
                        "required": ["pm2_5", "pm10"],
                    },
                    "timestamp": {"type": "string", "format": "date-time"},
                },
                "required": ["city", "aqi", "category", "pollutants", "timestamp"],
            },
            "error_schema": {
                "type": "object",
                "properties": {
                    "error_code": {"type": "string", "enum": ["LOCATION_NOT_FOUND", "SENSOR_OFFLINE", "UPSTREAM_TIMEOUT"]},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                },
                "required": ["error_code", "message", "retryable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Retrieve current meteorological conditions including temperature, relative humidity, wind speed, and computed heat index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the target city or district.",
                    },
                    "lat": {
                        "type": "number",
                        "description": "Latitude coordinate of the query location.",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude coordinate of the query location.",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temperature_celsius": {"type": "number"},
                    "relative_humidity_percent": {"type": "number", "minimum": 0, "maximum": 100},
                    "heat_index_celsius": {"type": "number"},
                    "wind_speed_kmh": {"type": "number"},
                    "condition": {"type": "string", "description": "Summary weather state (e.g., 'Sunny', 'Hazy', 'Overcast')."},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
                "required": ["city", "temperature_celsius", "relative_humidity_percent", "heat_index_celsius", "timestamp"],
            },
            "error_schema": {
                "type": "object",
                "properties": {
                    "error_code": {"type": "string", "enum": ["WEATHER_SERVICE_UNAVAILABLE", "INVALID_COORDINATES"]},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                },
                "required": ["error_code", "message", "retryable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate response text into the user's preferred language and script (e.g., Urdu in Nastaliq or Roman Urdu).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Source text to translate.",
                    },
                    "lang": {
                        "type": "string",
                        "enum": ["ur", "roman_ur", "en", "sd"],
                        "description": "Target language code: 'ur' (Urdu script), 'roman_ur' (Roman Urdu), 'en' (English), 'sd' (Sindhi).",
                    },
                },
                "required": ["text", "lang"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "original_text": {"type": "string"},
                    "translated_text": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["original_text", "translated_text", "target_language"],
            },
            "error_schema": {
                "type": "object",
                "properties": {
                    "error_code": {"type": "string", "enum": ["UNSUPPORTED_LANGUAGE", "TRANSLATION_FAILED"]},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                },
                "required": ["error_code", "message", "retryable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_health_advice",
            "description": "Search official municipal and medical knowledge base for heatwave and air pollution health advisories matching vulnerability profile and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query describing the demographic vulnerability, exposure context, or environmental severity (e.g., 'asthma patient outdoors during high PM2.5').",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "advisories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "category": {"type": "string", "enum": ["heatwave", "air_quality", "general_health"]},
                                "target_group": {"type": "string", "description": "Target population such as 'children', 'elderly', 'asthma', 'outdoor_workers'."},
                                "guidance": {"type": "string", "description": "Actionable, medically-vetted health advice string."},
                                "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["id", "category", "guidance", "relevance_score"],
                        },
                    },
                    "total_matches": {"type": "integer"},
                },
                "required": ["advisories", "total_matches"],
            },
            "error_schema": {
                "type": "object",
                "properties": {
                    "error_code": {"type": "string", "enum": ["KB_UNAVAILABLE", "NO_RELEVANT_MATCH"]},
                    "message": {"type": "string"},
                    "retryable": {"type": "boolean"},
                },
                "required": ["error_code", "message", "retryable"],
            },
        },
    },
]

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    schema["function"]["name"]: schema for schema in TOOL_SCHEMAS
}
