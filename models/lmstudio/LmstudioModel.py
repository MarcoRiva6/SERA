from lmstudio import LMStudioError

from models.model import Model
from queries.test import Query


class LmstudioModel(Model):

    def _submit_direct_inline(self, prompt: str) -> str:
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

    def _submit_lotus_inline(self, prompt, df):
        import lotus
        from lotus.cache import CacheFactory, CacheConfig, CacheType
        from lotus.models import LM

        cache_config = CacheConfig(cache_type=CacheType.SQLITE, max_size=1000)
        cache = CacheFactory.create_cache(cache_config)

        lm = LM(model="lm_studio/" + self.name_api,
                api_base="http://127.0.0.1:1234/v1",
                max_tokens=self.max_tokens if self.max_tokens else None,
                cache=cache,
                temperature=0.6)

        lotus.settings.configure(lm=lm, enable_cache=True)

        return prompt(df)

    def _submit_direct(self, queries: list[Query]) -> None:
        for query in queries:
            query.response = self._submit_direct_inline(query.prompt)

    def _submit_lotus(self, queries: list[Query], df) -> None:
        for query in queries:
            query.response = self._submit_lotus_inline(query.prompt, df)