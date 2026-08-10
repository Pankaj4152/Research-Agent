from google.genai import types

# -------------------------
# Custom Python Functions
# -------------------------

def get_weather(city: str) -> str:
    """Temporary fake weather data function."""
    weather_data = {
        "jodhpur": "35°C, sunny",
        "delhi": "32°C, cloudy",
        "mumbai": "29°C, rainy",
    }
    return weather_data.get(city.lower(), "Weather data not available")


def get_population(city: str) -> str:
    """Temporary fake population data function."""
    population_data = {
        "jodhpur": "1.5 million",
        "delhi": "33 million",
        "mumbai": "21 million",
    }
    return population_data.get(city.lower(), "Population data not available")


# -------------------------
# Gemini Tool Schemas
# -------------------------

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get the current weather for a city",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(
                            type="STRING",
                            description="Name of the city"
                        )
                    },
                    required=["city"]
                )
            ),
            types.FunctionDeclaration(
                name="get_population",
                description="Get the population of a city",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(
                            type="STRING",
                            description="Name of the city"
                        )
                    },
                    required=["city"]
                )
            )
        ]
    )
]

# Map tool names to Python functions
tool_map = {
    "get_weather": get_weather,
    "get_population": get_population,
}


def execute_tool(function_call) -> str:
    """Execute a function call requested by the Gemini model."""
    tool_name = function_call.name
    arguments = function_call.args

    tool = tool_map.get(tool_name)
    if tool is None:
        return "Unknown tool"

    return tool(**arguments)
