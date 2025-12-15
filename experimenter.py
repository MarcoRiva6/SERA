import importlib
import importlib.util
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml
from huggingface_hub.hub_mixin import DataclassInstance
from pandas import DataFrame

from experiments.experiment import (Experiment)
from experiments.run_type import (RunType)
from queries.test import Query, queries_to_df

parent = Path(__file__).resolve().parent
models_folder: Path = parent / 'models'
query_folder: Path = parent / 'queries'
runs_folder: Path = parent / 'runs'

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
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print("Error YAML parsing experiment file:", test_file, exc)
                continue
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
                    test_family, test_name_path = q_name_split[0], q_name_split[1]
                    inner_run_folder = this_run_folder / test_family / test_name_path
                    test_base_args = {
                        'family': test_family,
                        'name_path': test_name_path,
                        'run_type': rt,
                        'run_folder': inner_run_folder / 'data',
                    }
                    q_class_path = 'queries.' + q['name'].replace('/', '.') + '.' + test_name_path
                    q_class = class_from_path(q_class_path)

                    #instanciate test parameters object
                    q_param_class = getattr(q_class, 'Params')
                    q_external_params = {k: v for k, v in q.items() if k != 'name'}
                    test_params = q_param_class(seed=exp_seed, **q_external_params)
                    # instantiate test object
                    test_obj = q_class(**test_base_args, params=test_params)

                    # instantiate model object
                    model_name_split = m['name'].split('/')
                    model_family = model_name_split[0]
                    model_class_path = 'models.' + model_family + '.' + model_family.capitalize() + 'Model' + '.' + model_family.capitalize() + 'Model'
                    model_class = class_from_path(model_class_path)
                    model_name_path = model_name_split[1]
                    model_base_args = {
                        'name_path': model_name_path,
                        'family': model_family,
                        'file': models_folder / model_family / (model_name_path + '.yaml'),
                        'run_type': rt,
                        'run_folder': inner_run_folder / 'results' / model_name_path / rt,
                    }
                    # instantiate model parameters object
                    model_param_class = getattr(model_class, 'Params')
                    model_external_args = {k: v for k, v in m.items() if k != 'name'}
                    model_params = model_param_class(seed=exp_seed, **model_external_args)
                    model = model_class.from_yaml_file(**model_base_args, params=model_params)
                    # instantiate experiment object
                    experiment = Experiment(
                        name=f"{test_file.stem}: {model.name}, {rt}, {test_family} - {test_obj.name}",
                        run_folder=this_run_folder,
                        model=model,
                        run_type=rt,
                        test=test_obj
                    )
                    results.append(experiment)
    return results


def merge_many_dataframes(
        dfs: Sequence[pd.DataFrame],
        commons: Iterable[str],
        separate: Iterable[str],
        prefixes: Sequence[str],
) -> pd.DataFrame:
    """
    Merge a variable number of dataframes column-wise.

    - `commons`: columns that are shared and should appear only once (taken from the first df).
    - `separate`: columns that should be kept distinct for each df, with suffixes.
    - columns not listed in either `commons` or `separate` are discarded.
    - assumes all dfs have the same index and are row-aligned.
    """

    if not dfs:
        raise ValueError("You must provide at least one DataFrame")

    commons = list(commons)
    separate = list(separate)

    n = len(dfs)

    if len(prefixes) != n:
        raise ValueError("Length of suffixes must match number of dataframes")

    # --- Common columns: take only from the first df ---
    first = dfs[0]
    missing_common = [c for c in commons if c not in first.columns]
    if missing_common:
        raise KeyError(f"Commons columns missing in first dataframe: {missing_common}")

    result = first[commons].copy()

    # --- Separate columns: one per df with prefix ---
    for suffix, df in zip(prefixes, dfs):
        # Check columns exist
        missing_sep = [c for c in separate if c not in df.columns]
        if missing_sep:
            raise KeyError(f"Separate columns missing in dataframe with suffix {suffix}: {missing_sep}")

        for col in separate:
            result[f"{suffix}_{col}"] = df[col].values

    return result

def get_object_attributes_names(obj) -> list[str]:
    return [f.name for f in fields(obj)]

def rename_selected_keys(d: dict, keys_to_change: list[str], prefix: str):
    """Return a *new* dictionary where selected keys are renamed."""
    keys_to_change = set(keys_to_change)

    return {
        (prefix + '_' + k) if k in keys_to_change else k: v
        for k, v in d.items()
    }

def merge_queries(queries: list[list[Query]], names: list[str]) -> DataFrame:
    q_attributes: list[str] = get_object_attributes_names(queries[0][0])
    to_drop = ['ground_truth', 'response_json_schema']
    merge_on = [attr for attr in q_attributes if attr not in ['evaluations', 'response','parsing_failed','ground_truth'] + to_drop]
    evaluation_names: list[str] = list(queries[0][0].evaluations.keys())
    to_separate = evaluation_names + ['response','parsing_failed']

    result: DataFrame = None
    for qs, name in zip(queries, names):
        rename_dict = {k: f"{k}_{name}" for k in to_separate}
        q_df = queries_to_df(qs).rename(columns=rename_dict)
        #drop list columns, needed for merging DF since they are not hashable
        list_columns = [
            col for col in q_df.select_dtypes(include=["object"]).columns
            if q_df[col].map(lambda x: isinstance(x, list)).any()
        ]
        q_df = q_df.drop(columns=to_drop + list_columns)

        if result is None:
            result = q_df
        else:
            result = pd.merge(result, q_df, on=[c for c in merge_on if c not in list_columns], how='outer', validate='one_to_one')
    return result


if __name__ == "__main__":
    experiments = load_experiments()
    if not experiments:
        print("No (valid) experiments found.")
        exit(1)
    a = input(f"Produced {len(experiments)} experiments:{[e.name for e in experiments]}\nPress Enter to continue...")
    if a != '':
        exit(0)
    unique_test_names = set([e.test.name_path for e in experiments])

    for test_name in unique_test_names:
        matches = [e for e in experiments if e.test.name_path == test_name]
        for exp in matches:
            if (exp.inner_folder / 'queries.parquet').exists():
                exp.display()
                print(f"Skipping experiment (already executed): {exp.name}")
                continue
            print(f"Running experiment: {exp.name}")
            exp.execute()
            print('\n')

        evaluated_matches = [e for e in matches if e.test.queries and all(q.evaluations is not None for q in e.test.queries)]
        if len(evaluated_matches) <= 1 or len(set([e.test.run_type for e in evaluated_matches])) != 1:
            print('\n')
            continue
        print(f"aggregating results for test: {test_name}")
        merged_df = merge_queries([e.test.queries for e in evaluated_matches], [e.model.name_path for e in evaluated_matches])
        merged_df.to_csv(evaluated_matches[0].test.run_folder.parent / 'results' / 'aggregated_queries.csv', index=False, decimal=',', sep=';')
        print('\n')
