import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
import yaml

from models.model import Model
from experiments.experiment import (Experiment)
from experiments.run_type import (RunType)

runs_folder: Path = Path('runs')

def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_experiments() -> list[Experiment]:
    results = []

    for test_file in Path('experiments').glob('*'):
        if test_file.is_dir() or test_file.name == 'experiment.py' or '__pycache__' in test_file.parts or '.DS_Store' in test_file.parts or 'run_type.py' in test_file.parts:
            continue

        with open(test_file) as f:
            data = yaml.safe_load(f)
        models = []
        raw_model = data.get('model', '*')
        for mf in Path('models').glob(raw_model + '.yaml'):
            if mf.is_dir() or mf.name == 'model.py' or mf.name == 'global.yaml' or '__pycache__' in mf.parts or '.DS_Store' in mf.parts:
                continue
            models.append(Model.from_yaml_file(mf))

        run_types = []
        raw_run_types = data.get('run_types', '*')
        if raw_run_types == '*':
            for run_type in RunType:
                run_types.append(run_type)
        else:
            run_types.append(RunType(raw_run_types))

        queries = []
        raw_queries = data.get('queries', '*')
        if raw_queries == '*':
            raw_queries = '**/*'
        for q in Path('queries').glob(raw_queries):
            if q.is_dir() or q.name == 'test.py' or '__pycache__' in q.parts or '.DS_Store' in q.parts or '__init__.py' in q.parts:
                continue

            class_path = '.'.join(q.with_suffix('').parts) + '.Main'
            cls = class_from_path(class_path)
            family = q.parts[-2]
            queries.append(cls(family, q.stem))

        this_run_folder = runs_folder / test_file.stem
        for mf in models:
            for rt in run_types:
                for q in queries:
                    experiment = Experiment(
                        name=f"{mf.name}, {rt}, {q.family}, {q.name}",
                        run_folder=this_run_folder,
                        model=replace(mf),
                        run_type=rt,
                        test=replace(q)
                    )
                    experiment.test.run_type = rt
                    experiment.model.run_type = rt
                    inner_run_folder = this_run_folder / experiment.test.family / experiment.test.name_path
                    experiment.test.run_folder = inner_run_folder / 'data'
                    experiment.model.run_folder = inner_run_folder / 'results' / experiment.model.name_path / rt
                    results.append(experiment)
    return results

if __name__ == "__main__":
    experiments = load_experiments()
    #for e in experiments:
    #    print(e)
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        exp.test.run_folder.mkdir(parents=True, exist_ok=True)
        exp.execute()
