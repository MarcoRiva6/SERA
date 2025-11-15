from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pandas as pd

import yaml
from pandas import DataFrame

data_folder = Path('data')

@dataclass
class Submission(ABC):
    prompt: str
    response: str
    evaluation: dict

@dataclass
class Query(ABC):
    family: str
    name: str
    name_path: str
    run_folder: Path = None
    debug: bool = True
    pre_submissions_df: DataFrame = None
    submissions: list[Submission] = None

    @classmethod
    def from_yaml_file(cls, file):
        with open(file) as f:
            return cls(**yaml.load(f, Loader=yaml.FullLoader), name_path=file.stem)

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def evaluate_submission(self, submission: Submission):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def submissions_to_csv(self, file_path: str = None):
        if file_path is None:
            file_path = self.run_folder / 'submissions.csv'
        rows = []

        for sub in self.submissions:
            base = asdict(sub)                # convert dataclass to dictionary
            eval_dict = base.pop("evaluation")  # remove evaluation dict
            flat = {**base, **eval_dict}      # flatten into top-level
            rows.append(flat)

        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)