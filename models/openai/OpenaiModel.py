from dataclasses import dataclass
from typing import Any
from openai import OpenAI
from pydantic import BaseModel

from models.model import BatchableModel, BatchableModelParams

type QIID = str

@dataclass
class OpenaiModel(BatchableModel[QIID, Any, BatchableModelParams]):
    client = None
    batch_max_tokens: int = 0
    supports_batched: bool = True
    reasoning: bool = False

    def _init_model(self) -> None:
        self.client = OpenAI()

    def _submit_prompt(self, prompt: str, schema: BaseModel) -> tuple[str, int]:
        response = self.client.responses.create(
            model=self.name_api,
            input=prompt,
            
        )

        return response.output_text, response.usage.total_tokens if response.usage is not None else 0