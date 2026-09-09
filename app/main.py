import os
import uuid

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types

# -------------------------
# Gemini setup
# -------------------------

load_dotenv(override=True)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

from app.tools import execute_tool, tools

# -------------------------
# Session Store
# -------------------------

sessions: dict[str, list[types.Content]] = {}


def get_session_history(session_id: str) -> list[dict]:
    """Retrieve serializable history for a session."""
    contents = sessions.get(session_id, [])
    history = []
    for content in contents:
        parts_text = []
        for part in content.parts:
            if part.text:
                parts_text.append(part.text)
            elif part.function_call:
                parts_text.append(f"[Tool Call: {part.function_call.name}]")
            elif part.function_response:
                parts_text.append(f"[Tool Result: {part.function_response.name}]")
        history.append({
            "role": content.role,
            "content": "\n".join(parts_text)
        })
    return history


def clear_session(session_id: str) -> bool:
    """Clear memory for a given session."""
    if session_id in sessions:
        del sessions[session_id]
        return True
    return False


# -------------------------
# Agent Loop
# -------------------------

def ask_agent(prompt: str, session_id: str | None = None) -> tuple[str, str]:
    """Execute prompt using Gemini Flash with tool execution and multi-turn session memory."""
    if not session_id:
        session_id = str(uuid.uuid4())

    contents = sessions.get(session_id, [])

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        )
    )

    while True:
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
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
            return "Could not get a response from the model.", session_id

        # Save Gemini's complete response
        if not response.candidates:
            print(
                "\nEmpty response. Finish reason:",
                response.model_dump(
                    exclude_none=True
                ).get('candidates')
            )
            return "The model returned no response.", session_id

        model_content = response.candidates[0].content
        contents.append(model_content)

        # Find all custom function calls
        function_calls = [
            part.function_call
            for part in model_content.parts
            if part.function_call
        ]

        # No custom function calls. Answer complete.
        if not function_calls:
            sessions[session_id] = contents
            return response.text, session_id

        # Execute every custom function requested by Gemini
        tool_parts = []
        for function_call in function_calls:
            print("\n[Agent Tool Call]:", function_call.name, "Args:", function_call.args)
            result = execute_tool(function_call)
            print("[Tool Output]:", result)

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
# Main CLI Loop
# -------------------------

if __name__ == "__main__":
    current_session_id = str(uuid.uuid4())
    print(f"--- Research Agent Interactive Session ({current_session_id[:8]}) ---")
    print("Type 'exit' to quit or 'clear' to reset chat memory.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            if prompt.lower() == "exit":
                print("Goodbye!")
                break
            if prompt.lower() == "clear":
                clear_session(current_session_id)
                current_session_id = str(uuid.uuid4())
                print("Session memory cleared.\n")
                continue

            answer, current_session_id = ask_agent(prompt, session_id=current_session_id)
            print(f"\nAgent: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break