import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API key loaded:", api_key is not None)
print("API key length:", len(api_key) if api_key else 0)

client = genai.Client(api_key=api_key)


def ask_llm(prompt):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    prompt = "Write a short poem about the beauty of nature."

    answer = ask_llm(prompt)

    print(answer)