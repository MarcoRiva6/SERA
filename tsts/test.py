from dataclasses import dataclass, asdict
from pathlib import Path

from models.model import Model
from queries.query import Query
from enum import Enum
import yaml

class RunMode(str, Enum):
    DIRECT = 'direct'
    LOTUS = 'lotus'

@dataclass
class Test:
    name: str
    run_folder: Path
    model: Model
    run_mode: RunMode
    query: Query

    def execute(self):
        self.query.prepare()

        for i, s in enumerate(self.query.submissions):
            print("Running submission", i+1, '/', len(self.query.submissions))
            response = self.model.submit(s.prompt)
            s.response = response
            print("Received response:", s.response)
            s.evaluation = self.query.evaluate_submission(s)
            print('Evaluation:', s.evaluation)

        self.query.submissions_to_csv()

        print('test execution complete.')
