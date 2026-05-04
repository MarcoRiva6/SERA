from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from models.model import Model, SubmissionError
from queries.test import Test

from experiments.run_type import RunType

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    inner_folder: Path = None
    skip_model: bool = False

    def __post_init__(self):
        self.inner_folder = self.run_folder / self.test.family / self.test.name_path / 'results' / self.model.name_path / self.run_type

    def execute(self) -> None:
        """
        Execute the experiment by preparing the test, submitting queries to the model and saving results.
        """
        test_has_already_run_once = (self.test.run_folder / self.test.params_file_name).exists()
        model_has_already_run_once = (self.model.run_folder / self.model.params_file_name).exists()
        if test_has_already_run_once and not self.test.same_params() or model_has_already_run_once and not self.model.same_params():
            print('Test parameters have changed since last execution. Stopping to avoid inconsistencies')
            return

        print('Preparing test...')
        self.test.save_params()
        self.model.save_params()
        queries_file_exists: bool = (self.test.run_folder / self.test.prepared_queries_file_name).exists()
        if queries_file_exists:
            print('Restoring already generated queries...')
            self.test.restore_queries()
        else:
            print('Generating queries...')
            self.test.generate_queries()
            print('Storing queries...')
            self.test.store_queries()
            self.test.queries_to_csv('prepared_queries.csv')
        print('test ready.')
        if self.skip_model:
            print('Skipping model...')
        else:
            try:
                self.model.run(self.test.queries, self.test)
                print('Model processing complete.')
            except NotImplementedError as e:
                print(e)
                return
            except SubmissionError as e:
                print('Submission error:', e)
            except Exception as e:
                print('An unknown error occurred during submission:', e)
                return

        # facciamo partire il parsing e la valutazione delle query solo se ce n'è almeno una con risposta.
        # Poi, verranno valutate tutte le query, anche quelle senza risposta, le quali risulteranno quindi parsing_failed = True
        at_least_one_completed = any(q.is_answered() for q in self.test.queries)
        if at_least_one_completed:
            for q in tqdm(self.test.queries, desc='Parsing and evaluating queries', unit='query', colour='blue'):
                if q.is_answered():
                    self.test.parse_query(q)
                    q.evaluations = self.test.evaluate_query(q)

            print('Storing answered queries...')
            self.test.queries_to_csv('evaluated_queries.csv', self.inner_folder)
            self.test.queries_to_pickle(self.inner_folder / 'evaluated_queries.pkl')
            print('queries saved to csv and pickle.')
            self.test.evaluate()
            print('test evaluation:', self.test.evaluations)
            self.test.evaluations_to_csv(dest=self.inner_folder)
            print('evaluations saved to csv.')

        print('test execution completed.')
