import datetime
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from pydantic_core import TzInfo
from together import Together
if not load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env"):
    raise FileNotFoundError()
from together import Together

client = Together()

# Cancel a specific batch by ID
#batch_id = "2c662034-9db6-4cb8-93b3-2315458be989"
#cancelled_batch = client.batches.cancel_batch(batch_id)
#print(cancelled_batch)

batches = client.batches.list_batches()

for batch in batches:
    print(batch)

exit()
error_path = Path(__file__).parent.parent / "batch_error.json"
with error_path.open("w", encoding="utf-8") as f:
    f.write(status.model_dump_json())