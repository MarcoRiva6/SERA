from dataclasses import dataclass, asdict
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
    model: Model
    run_mode: RunMode
    query: Query

    def dest(self):
        return self.model.name_path + '/' + self.run_mode + '/' + self.query.family

    def run(self):
        self.query.prepare()

        for s in self.query.submissions:
            print("Running submission")
            response = self.model.submit(s.prompt)
            s.response = response
            print("Received response:", s.response)
