from datetime import datetime
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import yaml

from models.model import Model
from tsts.test import Test
from tsts.run_mode import RunMode

runs_folder = Path('runs')

def build_run_folder() -> Path:
    name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    run_folder = runs_folder / name
    return run_folder

def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_tests() -> list[Test]:
    results = []

    for file in Path('tsts').glob('*'):
        if file.is_dir() or file.name == 'test.py' or '__pycache__' in file.parts or '.DS_Store' in file.parts or 'run_mode.py' in file.parts:
            continue

        with open(file) as f:
            data = yaml.safe_load(f)
        models = []
        raw_model = data.get('model', '*')
        for mf in Path('models').glob(raw_model):
            if mf.is_dir() or mf.name == 'model.py' or mf.name == 'global.yaml' or '__pycache__' in mf.parts or '.DS_Store' in mf.parts:
                continue
            models.append(Model.from_yaml_file(mf))

        run_modes = []
        raw_run_modes = data.get('run_mode', '*')
        if raw_run_modes == '*':
            for run_mode in RunMode:
                run_modes.append(run_mode)
        else:
            run_modes.append(RunMode(raw_run_modes))

        queries = []
        raw_queries = data.get('queries', '*')
        if raw_queries == '*':
            raw_queries = '**/*'
        for q in Path('queries').glob(raw_queries):
            if q.is_dir() or q.name == 'query.py' or '__pycache__' in q.parts or '.DS_Store' in q.parts or '__init__.py' in q.parts:
                continue

            class_path = '.'.join(q.with_suffix('').parts) + '.Main'
            cls = class_from_path(class_path)
            family = q.parts[-2]
            queries.append(cls(family, q.stem))

        this_run_folder = build_run_folder()
        for mf in models:
            for rm in run_modes:
                for q in queries:
                    test = Test(
                        name=f"{mf.name}, {rm}, {q.family}, {q.name}",
                        run_folder=this_run_folder,
                        model=replace(mf),
                        run_mode=rm,
                        query=replace(q)
                    )
                    test.query.run_folder = this_run_folder / test.model.name_path / test.run_mode / test.query.family
                    results.append(test)
    return results

if __name__ == "__main__":
    tests = load_tests()
    for t in tests:
        print(t)
    for t in tests:
        print(f"Running test: {t.name}")
        t.query.run_folder.mkdir(parents=True, exist_ok=True)
        t.execute()
