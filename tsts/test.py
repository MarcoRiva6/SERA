from dataclasses import dataclass
from pathlib import Path

from models.model import Model
from queries.query import Query

from tsts.run_type import RunType

@dataclass
class Test:
    name: str
    run_folder: Path
    model: Model
    run_type: RunType
    query: Query

    def execute(self):
        self.query.prepare(self.run_type)

        for i, s in enumerate(self.query.submissions):
            print("Running submission", i+1, '/', len(self.query.submissions))
            response = self.model.submit(self.run_type, s.prompt, self.query.full_df)
            s.response = response
            print("Received response:", s.response)
            s.evaluations = self.query.evaluate_submission(s, self.run_type)
            print('Submission Evaluation:', s.evaluations)

        self.query.submissions_to_csv()
        self.query.evaluate()
        print('Test evaluations:', self.query.evaluations)
        self.query.evaluations_to_csv()

        print('test execution complete.')
