from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

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
    inner_folder: Path = None
    skip_model: bool = False

    def __post_init__(self):
        self.inner_folder = self.run_folder / self.test.family / self.test.name_path / 'results' / self.model.name_path / self.run_type

    def execute(self) -> None:
        """
        Execute the experiment by preparing the test, submitting queries to the model and saving results.
        """
        print('Preparing test...')
        self.test.init_queries_registry()
        needs_query_generation = False
        # è il nuovo equivalente di test_has_already_run_once
        queries_file_exists: bool = (self.test.run_folder / self.test.prepared_queries_file_name).exists()
        if queries_file_exists:
            if not self.test.same_params():
                if not self.test.compatible_params():
                    print('Test parameters have changed since last execution and are not compatible with the new ones. terminating...')
                    return
                needs_query_generation = True
            print('Restoring already generated queries...')
            self.test.restore_queries()
        else:
            needs_query_generation = True
        if needs_query_generation:
            print('Generating queries...')
            try:
                self.test.generate_queries()
            except NotImplementedError as e:
                print(e)
                return
            self.test.store_queries()
            self.test.queries_to_csv('prepared_queries.csv')
            self.test.save_params()
        print('test ready.')

        if self.skip_model:
            print('Skipping model...')
        else:
            model_has_already_run_once = (self.model.run_folder / self.model.params_file_name).exists()
            if model_has_already_run_once and not self.model.same_params():
                print('Model parameters have changed since last execution.')
                return
            self.model.save_params()
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
