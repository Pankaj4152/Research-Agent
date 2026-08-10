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


from app.tools import execute_tool, tools


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
            print("\nGemini API error:", error.code, "-", error.message)
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