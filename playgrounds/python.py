import json
from pathlib import Path

output_path = Path('/Users/***REMOVED***/Projects/similarity/runs/first-test/purchases/customer_segmentation/results/deepseek-r1-together/direct/batch_output.jsonl')

total_sum = 0
with output_path.open("r", encoding="utf-8") as f:
    for line in f:
        response = json.loads(line)
        total_sum += response['response']['body']['usage']['total_tokens']

print("Total tokens used:", total_sum)