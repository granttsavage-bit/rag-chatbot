from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: {api_key[:10] if api_key else 'NOT FOUND'}")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say hello and tell me you're working correctly."}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
    