from dataclasses import dataclass
from pathlib import Path

from models.model import Model
from queries.query import Query

from tsts.run_mode import RunMode

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
            response = self.model.submit(self.run_mode, s.prompt, self.query.full_df)
            s.response = response
            print("Received response:", s.response)
            s.evaluations = self.query.evaluate_submission(s, self.run_mode)
            print('Submission Evaluation:', s.evaluations)

        self.query.submissions_to_csv()
        self.query.evaluate()
        print('Test evaluations:', self.query.evaluations)
        self.query.evaluations_to_csv()

        print('test execution complete.')
