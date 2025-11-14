from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml
from pandas import DataFrame

data_folder = Path('data')
run_folder = Path('runs')

@dataclass
class Submission(ABC):
    prompt: str
    response: str

@dataclass
class Query(ABC):
    family: str
    name: str
    name_path: str
    debug: bool = True
    tests_df: DataFrame = None
    submissions: list[Submission] = None

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)

    @abstractmethod
    def prepare(self):
        pass
