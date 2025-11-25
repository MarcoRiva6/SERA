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
        if data.get('disabled', False):
            print("Warning: Skipping disabled experiment file:", test_file)
            continue
        exp_seed = data.get('seed', 0)

        models: list[Path] = []
        raw_models = data.get('models', '*')
        if not isinstance(raw_models, list):
            raw_models = [raw_models]
        for m in raw_models:
            for m_path in Path('models').glob(m + '.yaml'):
                if m_path.is_dir() or m_path.name == 'model.py' or m_path.name == 'global.yaml' or '__pycache__' in m_path.parts or '.DS_Store' in m_path.parts:
                    continue
                models.append(m_path)
        if not models:
            print("Warning: No models found for experiment file:", test_file)
            continue

        run_types: list[RunType] = []
        raw_run_types = data.get('run_types', '*')
        if raw_run_types == '*':
            for run_type in RunType:
                run_types.append(run_type)
        else:
            run_types.append(RunType(raw_run_types))
        if not run_types:
            print("Warning: No run types found for experiment file:", test_file)
            continue

        queries: list[Path] = []
        raw_queries = data.get('queries', '*')
        if raw_queries == '*':
            raw_queries = '**/*'
        else:
            raw_queries = f'**/{raw_queries}.py'
        for q_path in Path('queries').glob(raw_queries):
            if q_path.is_dir() or q_path.name == 'test.py' or '__pycache__' in q_path.parts or '.DS_Store' in q_path.parts or '__init__.py' in q_path.parts:
                continue
            queries.append(q_path)
        if not queries:
            print("Warning: No queries found for experiment file:", test_file)
            continue

        this_run_folder = runs_folder / test_file.stem
        for m_path in models:
            for rt in run_types:
                for q_path in queries:
                    family = q_path.parts[-2]
                    test_name_path = q_path.stem
                    inner_run_folder = this_run_folder / family / test_name_path
                    # build test object
                    class_path = '.'.join(q_path.with_suffix('').parts) + '.' + test_name_path
                    cls = class_from_path(class_path)
                    test = cls(family, test_name_path, rt, inner_run_folder / 'data', exp_seed)
                    # build model object
                    model = Model.from_yaml_file(m_path, rt, inner_run_folder, exp_seed)
                    experiment = Experiment(
                        name=f"{model.name}, {rt}, {family} - {test.name}",
                        run_folder=this_run_folder,
                        model=model,
                        run_type=rt,
                        test=test
                    )
                    results.append(experiment)
    return results

if __name__ == "__main__":
    experiments = load_experiments()
    if not experiments:
        print("No (valid) experiments found.")
    #for e in experiments:
    #    print(e)
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        exp.test.run_folder.mkdir(parents=True, exist_ok=True)
        exp.execute()
        print('\n')
