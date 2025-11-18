from dataclasses import dataclass, asdict
import yaml
from lmstudio import LMStudioError

from tsts.run_mode import RunMode


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

        lm = LM(model="lm_studio/" + self.name_api, api_base="http://127.0.0.1:1234/v1", max_tokens=self.max_tokens, cache=cache, temperature=0.6)

        lotus.settings.configure(lm=lm, enable_cache=True)

        return prompt(df)

    def submit(self, run_mode: RunMode, prompt, df=None):
        if run_mode == RunMode.DIRECT:
            return self.submit_lm_studio(prompt)
        elif run_mode == RunMode.LOTUS:
            return self.submit_lotus(prompt, df)