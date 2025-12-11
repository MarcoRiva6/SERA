import os
from pathlib import Path

from dotenv import load_dotenv


if not load_dotenv(dotenv_path=Path(__file__).parent / ".env"):
    raise FileNotFoundError()
api_key = os.getenv("GOOGLE_API_KEY")
from google.genai import Client
from google import genai
client =  Client(api_key=api_key)

batches = client.batches.list()

def print_batch(batch):
    print("─" * 80)
    print(f"Batch ID:       {batch.name}")
    print(f"Model:          {getattr(batch, 'model', None)}")
    print(f"Display Name:   {getattr(batch, 'display_name', None)}")
    print(f"State:          {batch.state}")
    print(f"Create Time:    {batch.create_time}")
    print(f"Update Time:    {batch.update_time}")
    print(f"Error:          {batch.error.message if getattr(batch, 'error', None) else None}")

    print("─" * 80)
    print()

for batch in batches:
    print_batch(batch)

