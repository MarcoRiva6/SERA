from dataclasses import dataclass, field

from lmstudio import LMStudioError

from models.model import Model
from queries.test import Query


class LmstudioModel(Model):

    address: str = "localhost:1234"
    @dataclass
    class Params(Model.Params):
        retries: int = 3
    params: Params = field(default_factory=Params)

    def _submit_direct_inline(self, prompt: str, format) -> str:
        import lmstudio as lms
        retries = self.params.retries

        while retries:
            try:
                with lms.Client(self.address) as client:
                    model = client.llm.model(self.name_api)

                    response = model.respond(
                        prompt,
                        response_format=format,
                        on_prompt_processing_progress = (lambda progress: print(f"{progress*100}% complete")))

                return response.content

            except LMStudioError as e:
                print("LM Studio exception submission:", e)
                retries -= 1
                import time
                time.sleep(5)

        raise Exception("Failed to get response from LM Studio after retries.")

    def _submit_lotus_inline(self, prompt, df):
        import lotus
        from lotus.cache import CacheFactory, CacheConfig, CacheType
        from lotus.models import LM

        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)

        lm = LM(model="lm_studio/" + self.name_api,
                api_base=f"http://{self.address}/v1",
                max_tokens=self.max_tokens if self.max_tokens else None,
                cache=cache,
                temperature=0.6)

        lotus.settings.configure(lm=lm, enable_cache=True)

        return prompt(df)

    def _submit_direct(self, queries: list[Query]) -> None:
        for query in queries:
            query.response = self._submit_direct_inline(query.prompt, query.response_json_schema)

    def _submit_lotus(self, queries: list[Query], df) -> None:
        for query in queries:
            query.response = self._submit_lotus_inline(query.prompt, df)