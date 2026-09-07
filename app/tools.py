import requests
from google.genai import types
from app.rag import search_local_docs

# -------------------------
# Real External API Tools
# -------------------------


def get_weather(city: str) -> str:
    """Fetch live real-time weather using Open-Meteo free API."""
    try:
        # Step 1: Get latitude and longitude of the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url, timeout=5)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find coordinates for city: '{city}'"

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_name = location.get("name", city)
        country = location.get("country", "")

        # Step 2: Fetch current weather for the coordinates
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()

        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")

        return f"Real-time weather in {city_name}, {country}: {temp}°C, wind speed {wind} km/h."
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


def search_wikipedia(query: str) -> str:
    """Search Wikipedia REST API for real summary information on any topic/person/city."""
    try:
        formatted_query = query.strip().replace(" ", "_")
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_query}"
        headers = {"User-Agent": "ResearchAgent/1.0"}
        
        response = requests.get(wiki_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            if extract:
                return extract
        
        return f"No detailed Wikipedia summary found for query: '{query}'."
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"


# -------------------------
# Gemini Tool Schemas
# -------------------------

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get live real-time weather information for any city in the world",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "city": types.Schema(
                            type="STRING",
                            description="Name of the city (e.g. Jodhpur, Tokyo, London)"
                        )
                    },
                    required=["city"]
                )
            ),
            types.FunctionDeclaration(
                name="search_wikipedia",
                description="Search Wikipedia for facts, summaries, population, and information on any topic, person, place, or concept",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Search query or topic to look up on Wikipedia (e.g. Jodhpur, Quantum Computing, Albert Einstein)"
                        )
                    },
                    required=["query"]
                )
            ),
            types.FunctionDeclaration(
                name="search_local_docs",
                description="Search internal knowledge base and local documents (e.g., RAG, architecture, concepts, internal knowledge)",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Search query to retrieve relevant paragraphs from internal vector document store"
                        )
                    },
                    required=["query"]
                )
            )
        ]
    )
]

# Map tool names to Python functions
tool_map = {
    "get_weather": get_weather,
    "search_wikipedia": search_wikipedia,
    "search_local_docs": search_local_docs,
}


def execute_tool(function_call) -> str:
    """Execute a function call requested by the Gemini model."""
    tool_name = function_call.name
    arguments = function_call.args

    tool = tool_map.get(tool_name)
    if tool is None:
        return "Unknown tool"

    return tool(**arguments)
