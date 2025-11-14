from dataclasses import dataclass, asdict
import yaml

def submit_lm_studio(model_name: str, prompt: str) -> str:
    import lmstudio as lms
    SERVER_API_HOST = "localhost:1234"

    with lms.Client(SERVER_API_HOST) as client:
        model = client.llm.model(model_name)

        response = model.respond(
            prompt,
            on_prompt_processing_progress = (lambda progress: print(f"{progress*100}% complete")))

    return response.content


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