from datetime import datetime
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import yaml

from models.model import Model
from experiments.experiment import (Experiment)
from experiments.run_type import (RunType)

runs_folder = Path('runs')

def build_run_folder() -> Path:
    name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    run_folder = runs_folder / name
    return run_folder

def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_experiments() -> list[Experiment]:
    results = []

    for file in Path('experiments').glob('*'):
        if file.is_dir() or file.name == 'experiment.py' or '__pycache__' in file.parts or '.DS_Store' in file.parts or 'run_type.py' in file.parts:
            continue

        with open(file) as f:
            data = yaml.safe_load(f)
        models = []
        raw_model = data.get('model', '*')
        for mf in Path('models').glob(raw_model):
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

        this_run_folder = build_run_folder()
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
                    experiment.test.run_folder = this_run_folder / experiment.model.name_path / experiment.run_type / experiment.test.family
                    experiment.test.run_type = rt
                    experiment.model.run_type = rt
                    results.append(experiment)
    return results

if __name__ == "__main__":
    experiments = load_experiments()
    for e in experiments:
        print(e)
    for e in experiments:
        print(f"Running experiment: {e.name}")
        e.test.run_folder.mkdir(parents=True, exist_ok=True)
        e.execute()
