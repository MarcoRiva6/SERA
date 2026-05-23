from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from together.types import BatchJobStatus

if not load_dotenv(dotenv_path=Path(__file__).parent / ".env"):
    raise FileNotFoundError('.env file not found')
from together import Together

client = Together()

raw_batches = client.batches.list_batches()
#client.batches.cancel_batch(batch_job_id="")

def _created_utc(batch):
    created = batch.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc)
batches = sorted(raw_batches, key=_created_utc, reverse=False)

local_tz = datetime.now().astimezone().tzinfo

for batch in batches:
    created = batch.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created_local = created.astimezone(local_tz)
    created_str = created_local.strftime("%Y-%m-%d %H:%M:%S")

    output = f"Batch ID: {batch.id} Created At: {created_str}"
    output += f" model: {batch.model_id}"
    output += f" status: {batch.status.value}"
    if batch.status == BatchJobStatus.IN_PROGRESS:
        output += f" progress: {batch.progress}"
    elif batch.status == BatchJobStatus.FAILED:
        output += f" error: {batch.error} error_file_id: {batch.error_file_id}"
    elif batch.status == BatchJobStatus.COMPLETED:
        completed_at = batch.completed_at
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        completed_local = completed_at.astimezone(local_tz)
        completed_str = completed_local.strftime("%Y-%m-%d %H:%M:%S")
        output += f" output_file_id: {batch.output_file_id} completed_at: {completed_str}"
    print(output)

exit()