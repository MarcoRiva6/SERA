from dataclasses import dataclass, asdict
import yaml
from lmstudio import LMStudioError

from experiments.run_type import RunType
from enum import StrEnum, auto

class Backend(StrEnum):
    LM_STUDIO = auto()
    TOGETHER = auto()

@dataclass
class Model:
    name: str
    name_path: str
    name_api: str
    backend: Backend
    max_tokens: int
    backend: str

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)

    def submit_lm_studio(self, prompt: str) -> str:
        import lmstudio as lms
        server_api_host = "localhost:1234"
        retries = 3

        while retries:
            try:
                with lms.Client(server_api_host) as client:
                    model = client.llm.model(self.name_api)

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

    def submit_lotus(self, prompt, df):
        import lotus
        from lotus.cache import CacheFactory, CacheConfig, CacheType
        from lotus.models import LM

        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)

        if self.backend == "lm_studio":
            model_str = "lm_studio/" + self.name_api
        else:
            model_str = self.name_api
        lm = LM(model=model_str, api_base="http://127.0.0.1:1234/v1", max_tokens=self.max_tokens if self.max_tokens else None, cache=cache, temperature=0.6)

        lotus.settings.configure(lm=lm, enable_cache=True)

        return prompt(df)

    def submit_together(self, prompt: str) -> str:
        from together import Together

        client = Together()

        return client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=False,
        ).response().choices[0].message.content

    def submit(self, run_type: RunType, prompt, df=None):
        if run_type == RunType.DIRECT:
            if self.backend == Backend.LM_STUDIO:
                return self.submit_lm_studio(prompt)
            elif self.backend == Backend.TOGETHER:
                return self.submit_together(prompt)
        elif run_type == RunType.LOTUS:
            return self.submit_lotus(prompt, df)
        else:
            raise NotImplementedError(f'Run mode {run_type} not implemented.')