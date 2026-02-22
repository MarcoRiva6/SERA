import json
from dataclasses import dataclass, field

from ollama import Client
from tqdm import tqdm

from models.model import Model
from queries.test import Query


@dataclass
class OllamaModel(Model):
    client: Client = None
    address: str = "localhost:11434"
    supports_thinking: bool = False
    @dataclass
    class Params(Model.Params):
        pass
    params: Params = field(default_factory=Params)

    def _init_model(self) -> None:
        self.client = Client(host="http://" + self.address)

    def _submit_direct(self, queries: list[Query]) -> None:
        file_path = self.run_folder / 'responses.jsonl'

        processed_queries = {}
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_queries[data['id']] = data['response']
            if len(processed_queries) > 0:
                print(f"Retrieved {len(processed_queries)} already processed answer{"s" if len(processed_queries) > 1 else ""}.")

        queries_to_process = []
        for query in queries:
            if query.id in processed_queries:
                query.response = processed_queries[query.id]
            else:
                queries_to_process.append(query)

        if not queries_to_process:
            return

        with open(file_path, 'a', encoding='utf-8') as f:
            for query in tqdm(queries_to_process, desc=f"Querying {self.name} via Ollama", unit="query", colour='yellow'):
                response = self.client.generate(
                    model=self.name_api,
                    prompt=query.prompt,
                    stream=False,
                    think=self.supports_thinking,
                    format=query.response_json_schema,
                )
                query.response = response.response

                record = {
                    "id": query.id,
                    "response": query.response
                }

                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()