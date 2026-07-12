import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url="https://api.fireworks.ai/inference/v1"
)
try:
    resp = client.chat.completions.create(
        model="minimax-m3",
        messages=[{"role": "user", "content": "hello"}]
    )
    print("SUCCESS with short name")
except Exception as e:
    print(f"FAILED with short name: {e}")

try:
    resp = client.chat.completions.create(
        model="accounts/fireworks/models/minimax-m3",
        messages=[{"role": "user", "content": "hello"}]
    )
    print("SUCCESS with full name")
except Exception as e:
    print(f"FAILED with full name: {e}")
