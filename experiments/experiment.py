from dataclasses import dataclass
from pathlib import Path
from models.model import Model, SubmissionError
from queries.test import Test

from experiments.run_type import RunType

@dataclass
class Experiment:
    """
    Represents an experiment. This includes the model to be tested, the type of run, and the test to be executed.
    """
    name: str
    run_folder: Path
    model: Model
    run_type: RunType
    test: Test

    def execute(self) -> None:
        """
        Execute the experiment by preparing the test, submitting queries to the model and saving results.
        """
        self.test.prepare_queries()
        print('test prepared. Submitting queries...')
        try:
            if not self.model.submit(self.test):
                print('Skipping the remaining part of the execution...')
                return
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
        inner_run_folder = self.run_folder / self.test.family / self.test.name_path / 'results' / self.model.name_path / self.run_type
        self.test.queries_to_csv('answered_queries.csv', inner_run_folder)
        print('queries saved to csv.')
        self.test.evaluate()
        print('test evaluation:', self.test.evaluations)
        self.test.evaluations_to_csv(dest=inner_run_folder)
        print('evaluations saved to csv.')

        print('test execution completed.')
