from dataclasses import dataclass
from pathlib import Path

from models.model import Model
from queries.test import Test

from experiments.run_type import RunType

@dataclass
class Experiment:
    name: str
    run_folder: Path
    model: Model
    run_type: RunType
    test: Test

    def execute(self):
        self.test.prepare()

        for i, q in enumerate(self.test.queries):
            print("Running query", i + 1, '/', len(self.test.queries))
            response = self.model.submit(q.prompt, self.test.full_df)
            q.response = response
            print("Received response:", q.response)
            q.evaluations = self.test.evaluate_query(q)
            print('Query Evaluation:', q.evaluations)

        self.test.queries_to_csv()
        self.test.evaluate()
        print('Test evaluations:', self.test.evaluations)
        self.test.evaluations_to_csv()

        print('test execution complete.')
