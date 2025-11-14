from pathlib import Path
import yaml
from models.model import Model
from queries.query import Query
from tsts.test import *
import importlib
import importlib.util
from types import ModuleType

def import_class(path: str):
    """
    Accepte 'package.module.ClassName' ou 'package.module:ClassName'.
    Retourne la classe.
    """
    if ":" in path:
        mod_name, cls_name = path.split(":", 1)
    else:
        mod_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)

def import_class_from_file(file_path: str, class_name: str):
    """
    Importe une classe depuis un fichier, par exemple `models/my_model.py`.
    """
    spec = importlib.util.spec_from_file_location("queries", file_path)
    module = importlib.util.module_from_spec(spec)  # type: ModuleType
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def load_tests() -> list[Test]:
    results = []

    for file in Path('tsts').glob('*'):
        if file.is_dir() or file.name == 'test.py' or '__pycache__' in file.parts or '.DS_Store' in file.parts:
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

            module = importlib.import_module('.'.join(q.with_suffix('').parts))
            cls = getattr(module, "Main")
            queries.append(cls(q.parts[-2], q.stem, q.stem))

        for mf in models:
            for rm in run_modes:
                for q in queries:
                    results.append(Test(
                        name=f"{mf.name}_{rm}_{q.family}_{q.family}_{q.name}",
                        model=mf,
                        run_mode=rm,
                        query=q
                    ))

    return results

if __name__ == "__main__":
    tests = load_tests()
    for t in tests:
        print(t)
    for t in tests:
        print(f"Running test: {t.name}")
        t.run()
