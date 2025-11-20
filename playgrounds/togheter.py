import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from together import Together
if not load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env"):
    raise FileNotFoundError()
api_key = os.getenv("TOGETHER_API_KEY")
poll_interval = 60 #seconds
timeout = 86400  #seconds (24 hours)
max_tokens = 300

client = Together(api_key=api_key)

batch_id = '5f0147d1-91a7-4f34-b63a-fbf3ab1e5253'
status = client.batches.get_batch(batch_id)
print(status)
output_path = Path(__file__).parent.parent / 'runs' / '25-11-19_23-53-45' / 'deepseek-r1-together' / 'direct' / 'movies' / "batch_output.jsonl"
with output_path.open("r", encoding="utf-8") as f:
    for line in f:
        response = json.loads(line)
        if response['response']['body']['choices'][0]['finish_reason'] == 'length':
            print(f"Warning: Response for query with prompt hash {response['id']} was cut off due to length.")
        print(response['response']['body']['choices'][0]['message']['content'])
#client.files.retrieve_content(id=status.output_file_id, output=str(output_path))
exit()
error_path = Path(__file__).parent.parent / "batch_error.json"
with error_path.open("w", encoding="utf-8") as f:
    f.write(status.model_dump_json())