from dataclasses import dataclass, asdict
from pathlib import Path

from models.model import Model
from queries.query import Query
from enum import StrEnum, auto
import yaml

class RunMode(StrEnum):
    DIRECT = auto()
    LOTUS = auto()

@dataclass
class Test:
    name: str
    run_folder: Path
    model: Model
    run_mode: RunMode
    query: Query

    def execute(self):
        if self.run_mode == RunMode.DIRECT:
            self.query.prepare_direct()
        elif self.run_mode == RunMode.LOTUS:
            self.query.prepare_lotus()

        for i, s in enumerate(self.query.submissions):
            print("Running submission", i+1, '/', len(self.query.submissions))
            response = self.model.submit(s.prompt)
            s.response = response
            print("Received response:", s.response)
            s.evaluations = self.query.evaluate_submission(s)
            print('Submission Evaluation:', s.evaluations)

        self.query.submissions_to_csv()
        self.query.evaluate()
        print('Test evaluations:', self.query.evaluations)
        self.query.evaluations_to_csv()

        print('test execution complete.')
