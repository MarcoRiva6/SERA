import importlib
import importlib.util
from dataclasses import replace, fields
from pathlib import Path
import yaml
from huggingface_hub.hub_mixin import DataclassInstance

from models.model import Model
from experiments.experiment import (Experiment)
from experiments.run_type import (RunType)

models_folder: Path = Path('models')
query_folder: Path = Path('queries')
runs_folder: Path = Path('runs')

def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def instantiate_from_yaml(cls, data: dict) -> DataclassInstance:
    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}

    return cls(**kwargs)

def load_experiments() -> list[Experiment]:
    results = []

    for test_file in Path('experiments').glob('*'):
        if test_file.is_dir() or test_file.name == 'experiment.py' or '__pycache__' in test_file.parts or '.DS_Store' in test_file.parts or 'run_type.py' in test_file.parts:
            continue

        with open(test_file) as f:
            data = yaml.safe_load(f)
        if data.get('disabled', False):
            print("Skipping disabled experiment file:", test_file)
            continue
        exp_seed = data.get('seed', 0)

        models: list[dict] = []
        queries: list[dict] = []

        for attr in [{'folder': models_folder, 'lst': models, 'attr_name': 'models', 'extension': '.yaml', 'exclude': ['model.py', 'global.yaml']},
                     {'folder': query_folder, 'lst': queries, 'attr_name': 'queries', 'extension': '.py', 'exclude': ['test.py']}]:
            pattern  = data.get(attr['attr_name'], '*')
            if pattern is None: # attr:
                print("Warning: No queries defined in experiment file:", test_file)
                continue
            elif pattern == '*': # attr: '*'
                for a_path in attr['folder'].glob('**/*' + attr['extension']):
                    if a_path.is_dir() or a_path.name in attr['exclude'] or '__init__.py' in a_path.parts:
                        continue
                    attr['lst'].append({'name': a_path.parts[-2] + '/' + a_path.stem})
            elif isinstance(pattern, list): # attr: [ ... ]
                for a in pattern:
                    if isinstance(a, dict): # attr: - name: 'some_val', ...
                        try:
                            a_name_path = a['name']
                        except KeyError:
                            print(f"Warning: dictionary {a} in {attr['attr_name']} missing 'name' key in experiment file:", test_file)
                            continue
                        a_path = attr['folder'] / (a_name_path + attr['extension'])
                        if not a_path.exists():
                            print(f"Warning: {attr['attr_name']} path does not exist:", a_path, "in experiment file:", test_file)
                            continue
                        attr['lst'].append(a)
                    elif isinstance(a, str): # attr: - 'some_val', ...
                        a_path = attr['folder'] / (a + attr['extension'])
                        if not a_path.exists():
                            print(f"Warning: {attr['attr_name']} path does not exist:", a_path, "in experiment file:", test_file)
                            continue
                        attr['lst'].append({'name': a})
                    else: # invalid format
                        print(f"Warning: Invalid format for {attr['attr_name']} in experiment file:", test_file)
                        continue
            elif isinstance(pattern, str): # attr: 'some_val'
                a_path = attr['folder'] / (pattern + attr['extension'])
                if not a_path.exists():
                    print(f"Warning: {attr['attr_name']} path does not exist:", a_path, "in experiment file:", test_file)
                    continue
                attr['lst'].append({'name': pattern})
            else: # invalid format
                print(f"Warning: Invalid format for {attr['attr_name']} in experiment file:", test_file)
                continue

            if len(attr['lst']) == 0:
                print(f"Warning: No matched {attr['attr_name']} for experiment file:", test_file)
                continue

        run_types: list[RunType] = []
        raw_run_types = data.get('run_types', '*')
        if raw_run_types is None:
            print("Warning: No run types defined in experiment file:", test_file)
            continue
        elif raw_run_types == '*':
            for run_type in RunType:
                run_types.append(run_type)
        else:
            run_types.append(RunType(raw_run_types))
        if not run_types:
            print("Warning: No run types matched for experiment file:", test_file)
            continue

        this_run_folder = runs_folder / test_file.stem
        for m in models:
            for rt in run_types:
                for q in queries:
                    # instantiate test object
                    q_name_split = q['name'].split('/')
                    family, test_name_path = q_name_split[0], q_name_split[1]
                    inner_run_folder = this_run_folder / family / test_name_path
                    q_class = class_from_path('queries.' + q['name'].replace('/', '.') + '.' + test_name_path)
                    q_extend = {
                        'family': family,
                        'name_path': test_name_path,
                        'run_type': rt,
                        'run_folder': inner_run_folder / 'data',
                        'seed': exp_seed
                    }
                    test_args = q_extend | {k: v for k, v in q.items() if k != 'name'}
                    test = q_class(**test_args)
                    # instantiate model object
                    model_name_path = m['name'].split('/')[1]
                    m_extend = {
                        'name_path': model_name_path,
                        'file': models_folder / (m['name'] + '.yaml'),
                        'run_type': rt,
                        'run_folder': inner_run_folder / 'results' / model_name_path / rt,
                        'seed': exp_seed
                    }
                    model_args = m_extend | {k: v for k, v in m.items() if k != 'name'}
                    model = Model.from_yaml_file(**model_args)
                    # instantiate experiment object
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
    #exit(0)
    for exp in experiments:
        if (exp.inner_folder / 'queries.parquet').exists():
            exp.display()
            print(f"Skipping experiment (already executed): {exp.name}")
            continue
        print(f"Running experiment: {exp.name}")
        exp.test.run_folder.mkdir(parents=True, exist_ok=True)
        exp.execute()
        print('\n')
