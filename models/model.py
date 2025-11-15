from dataclasses import dataclass, asdict
import yaml
from lmstudio import LMStudioError


def submit_lm_studio(model_name: str, prompt: str) -> str:
    import lmstudio as lms
    server_api_host = "localhost:1234"
    retries = 3

    while retries:
        try:
            with lms.Client(server_api_host) as client:
                model = client.llm.model(model_name)

                response = model.respond(
                    prompt,
                    on_prompt_processing_progress = (lambda progress: print(f"{progress*100}% complete")))

            return response.content

        except LMStudioError as e:
            print("LM Studio exception submission:", e)
            retries -= 1
            import time
            time.sleep(5)

    raise Exception("Failed to get response from LM Studio after retries.")


@dataclass
class Model:
    name: str
    name_path: str
    name_api: str
    backend: str
    max_tokens: int
    backend: str

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)

    def submit(self, prompt: str) -> str:
        if self.backend == "lm_studio":
            return submit_lm_studio(self.name_api, prompt)
        else:
            return ""