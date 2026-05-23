import os
from pathlib import Path

from dotenv import load_dotenv


if not load_dotenv(dotenv_path=Path(__file__).parent / ".env"):
    raise FileNotFoundError()
api_key = os.getenv("GOOGLE_API_KEY")
from google.genai import Client
from google import genai
client =  Client(api_key=api_key)

id = "batches/?"

client.batches.delete(name=id)

print('batch deleted')