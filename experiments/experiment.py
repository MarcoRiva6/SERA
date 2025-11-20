from dataclasses import dataclass
from pathlib import Path

from models.model import Model, SubmissionError
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
        print('test prepared. Submitting queries...')
        try:
            self.model.submit(self.test)
            print('test submit complete.')
        except NotImplementedError as e:
            print(e)
            return
        except SubmissionError as e:
            print('Submission error:', e)
            return
        except Exception as e:
            print('An unknown error occurred during submission:', e)
            return
        self.test.queries_to_csv('answered_queries.csv')
        print('queries saved to csv.')
        self.test.evaluate()
        print('test evaluation:', self.test.evaluations)
        self.test.evaluations_to_csv()
        print('evaluations saved to csv.')

        print('test execution completed.')
