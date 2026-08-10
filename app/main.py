import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types


# -------------------------
# Gemini setup
# -------------------------

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# -------------------------
# Custom tools
# -------------------------

def get_weather(city):
    # Temporary fake data.
    # Later this can become a real API call.
    weather_data = {
        "jodhpur": "35°C, sunny",
        "delhi": "32°C, cloudy",
        "mumbai": "29°C, rainy",
    }

    return weather_data.get(
        city.lower(),
        "Weather data not available"
    )


def get_population(city):
    # Temporary fake data.
    # Later this can come from a real database/API.
    population_data = {
        "jodhpur": "1.5 million",
        "delhi": "33 million",
        "mumbai": "21 million",
    }

    return population_data.get(
        city.lower(),
        "Population data not available"
    )


# -------------------------
# Tool definitions
# -------------------------

tools = [
    # Our custom Python tools
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


# -------------------------
# Map tool names to Python functions
# -------------------------

tool_map = {
    "get_weather": get_weather,
    "get_population": get_population,
}


# -------------------------
# Execute custom tool
# -------------------------

def execute_tool(function_call):

    tool_name = function_call.name
    arguments = function_call.args

    tool = tool_map.get(tool_name)

    if tool is None:
        return "Unknown tool"

    return tool(**arguments)


# -------------------------
# Agent
# -------------------------

def ask_agent(prompt):

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    ]

    while True:

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools,
                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
            )
        except errors.ClientError as error:

            print(
                "\nGemini API error:",
                error.code,
                "-",
                error.message
            )

            if error.code == 429:
                print(
                    "\nYou have exceeded the Gemini API quota. "
                    "Check your plan/billing at "
                    "https://ai.dev/rate-limit"
                )

            return "Could not get a response from the model."

        # Save Gemini's complete response
        if not response.candidates:
            print(
                "\nEmpty response. Finish reason:",
                response.model_dump(
                    exclude_none=True
                ).get('candidates')
            )
            return "The model returned no response."

        model_content = response.candidates[0].content

        contents.append(model_content)

        # Find all custom function calls
        function_calls = [
            part.function_call
            for part in model_content.parts
            if part.function_call
        ]

        # No custom function calls.
        # Gemini may have already completed the answer,
        # including using Google Search.
        if not function_calls:
            return response.text

        # Execute every custom function requested by Gemini
        tool_parts = []

        for function_call in function_calls:

            print(
                "\nTool requested:",
                function_call.name
            )

            print(
                "Arguments:",
                function_call.args
            )

            result = execute_tool(function_call)

            print(
                "Tool result:",
                result
            )

            tool_parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={
                        "result": result
                    }
                )
            )

        # Send ALL custom tool results back to Gemini
        contents.append(
            types.Content(
                role="tool",
                parts=tool_parts
            )
        )


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":

    prompt = input("Ask something: ")

    answer = ask_agent(prompt)

    print("\nFinal answer:")
    print(answer)