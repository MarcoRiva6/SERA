import importlib
import importlib.util
import inspect
from dataclasses import fields, asdict, dataclass
from pathlib import Path
from typing import get_origin, get_args, Any

import pandas as pd
import yaml
from pandas import DataFrame

from queries.test import Test

import models.model
from experiments.experiment import (Experiment)
from experiments.run_type import (RunType)
from queries.test import Query, queries_to_df

parent = Path(__file__).resolve().parent
models_folder: Path = parent / 'models'
query_folder: Path = parent / 'queries'
runs_folder: Path = parent / 'runs'

@dataclass
class Run:
    name_path: str
    run_folder: Path
    @dataclass
    class Query:
        name: str
        @dataclass
        class RunTypeExperiments:
            run_type: RunType
            experiments: list[Experiment]
            all_evaluated: bool = False
        run_type_experiments: list[RunTypeExperiments]
    queries: list[Query]
    seed: int = 0
    create_aggregated_csv: bool = False
    skip_model: bool = False

def get_subclasses(module_path, base_cls) -> list:
    module = importlib.import_module(module_path)
    return [
        cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base_cls)
           and cls is not base_cls
           and cls.__module__ == module.__name__
    ]

def get_test_class(modul_path):
    candidates = get_subclasses(modul_path, Test)
    if not candidates:
        raise Exception(f'No test class found for {modul_path}')
    if len(candidates) != 1:
        raise Exception(f'Found {len(candidates)} test classes for {modul_path}')
    return candidates[0]

def get_test_generic_types(test_cls: type):
    """
    Returns (T_Query, T_TestParameters, T_Evaluations) for a concrete Test subclass.
    """
    # Look at the generic base used in the class definition
    for base in getattr(test_cls, "__orig_bases__", ()):
        if get_origin(base) is Test:
            return get_args(base)

    raise TypeError(f"{test_cls.__name__} does not directly specify Test[...] generics")

def class_from_path(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_experiments() -> list[Run]:
    results: list[Run] = []

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

        test_file_name_path = test_file.stem
        this_run_folder = runs_folder / test_file_name_path
        r = Run(name_path=test_file_name_path, run_folder=this_run_folder, queries=[])
        results.append(r)

        try:
            r.seed = data['seed']
        except KeyError:
            pass
        try:
            r.create_aggregated_csv = data['create_aggregated_csv']
        except KeyError:
            pass
        try:
            r.skip_model = data['skip_model']
        except KeyError:
            pass


        models: list[dict] = []
        queries: list[dict] = []

        for attr in [{'folder': models_folder, 'lst': models, 'attr_name': 'models', 'extension': '.yaml', 'exclude': ['model.py', 'global.yaml']},
                     {'folder': query_folder, 'lst': queries, 'attr_name': 'queries', 'extension': '.py', 'exclude': ['test.py','metrics.py']}]:
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

        for q in queries:
            r_test = Run.Query(name=q['name'], run_type_experiments=[])
            r.queries.append(r_test)
            for rt in run_types:
                r_run_type_experiments = Run.Query.RunTypeExperiments(run_type=rt, experiments=[])
                r_test.run_type_experiments.append(r_run_type_experiments)
                for m in models:
                    # instantiate test object
                    q_name_split = q['name'].split('/')
                    test_family, test_name_path = q_name_split[0], q_name_split[1]
                    inner_run_folder = this_run_folder / test_family / test_name_path
                    test = instantiate_test(r.seed, inner_run_folder, q, rt, test_family, test_name_path)
                    test_dict = asdict(test)

                    model = instantiate_model(r.seed, inner_run_folder, m, rt)
                    # instantiate experiment object
                    experiment = Experiment(
                        name=f"{test_dict.get('name', test_dict.get('name_path'))}: {model.name} - {rt}",
                        run_folder=this_run_folder,
                        model=model,
                        run_type=rt,
                        test=test,
                        skip_model=r.skip_model
                    )
                    r_run_type_experiments.experiments.append(experiment)
    return results

def instantiate_test(exp_seed: int, inner_run_folder: Path, query_dictionary: dict, rt: RunType, test_family: str,
                     test_name_path: str) -> Test:
    test_base_args = {
        'family': test_family,
        'name_path': test_name_path,
        'run_type': rt,
        'run_folder': inner_run_folder / 'data' / rt,
    }
    q_module_path = 'queries.' + query_dictionary['name'].replace('/', '.')
    q_class = get_test_class(q_module_path)

    q_param_class = get_test_generic_types(q_class)[0]

    # instantiate test parameters object
    q_external_params = {k: v for k, v in query_dictionary.items() if k != 'name'}
    test_params = q_param_class(seed=exp_seed, **q_external_params)
    # instantiate test object
    test_obj = q_class(**test_base_args, parameters=test_params)
    return test_obj

def instantiate_model(exp_seed: int, inner_run_folder: Path, model_dictionary: dict, rt: RunType) -> models.model.Model:
    # instantiate model object
    model_name_split = model_dictionary['name'].split('/')
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
    model_external_args = {k: v for k, v in model_dictionary.items() if k != 'name'}
    model_params = model_param_class(seed=exp_seed, **model_external_args)
    model = model_class.from_yaml_file(**model_base_args, params=model_params)
    return model

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
    to_drop = ['response_json_schema', 'parameters', 'evaluations']
    evaluation_names: list[str] = list(asdict(queries[0][0].evaluations).keys())
    params_names: list[str] = list(asdict(queries[0][0].parameters).keys())
    to_separate = evaluation_names + ['response','parsed_response','parsing_failed']
    merge_on = [attr for attr in q_attributes + params_names if attr not in to_separate + to_drop]

    result: DataFrame = None
    for qs, name in zip(queries, names):
        q_df = queries_to_df(qs)
        #drop list columns, needed for merging DF since they are not hashable
        list_columns = [
            col for col in q_df.select_dtypes(include=["object"]).columns
            if q_df[col].map(lambda x: isinstance(x, list)).any()
        ]
        q_df = q_df.drop(columns=to_drop + list_columns, errors='ignore')
        # renaming evaluations to contain models names
        rename_dict = {k: f"{k}_{name}" for k in to_separate}
        q_df = q_df.rename(columns=rename_dict)

        if result is None:
            result = q_df
        else:
            result = pd.merge(result, q_df, on=[c for c in merge_on if c not in list_columns], how='outer', validate='one_to_one')
    return result

def prepare_for_charts(dicts: dict[str, dict[str, dict[str, list[Query]]]]) -> dict[str, dict[str, list[Query]]]:
    def _round_scalar(v):
        if isinstance(v, float):
            r = round(v / 0.05) * 0.05
            return round(r, 10)
        return v
    # run_type
    for_charts: dict[str, dict[str, list[Query]]] = {}
    for test_name, run_type_dict in dicts.items():
        for_charts[test_name] = {}
        for run_type, model_dict in run_type_dict.items():
            for model in model_dict.keys():
                try:
                    for_charts[test_name][model]
                except KeyError:
                    for_charts[test_name][model] = []
            for model_name, queries in model_dict.items():
                for q in queries:
                    setattr(q.parameters, 'run_type', run_type)
                for_charts[test_name][model_name].extend(queries)
    # stampa i failing rates
    for test_name, model_dict in for_charts.items():
        for model_name, queries in model_dict.items():
            print(f"Parsing failed rate for {test_name} - {model_name}: {sum(q.parsing_failed for q in queries)/len(queries) * 100:.2f}%")
    # raggruppiamo per dataset
    for test_name, model_dict in for_charts.items():
        for model_name, queries in model_dict.items():
            for q in queries:
                setattr(q.parameters, 'dataset', test_name.split('/')[0])
    # raggruppamento noti - ignoti
    noti = ('cities/global_liveability', 'planets/esi', 'purchases/customer_segmentation','molecules/levenshtein')
    for test_name, model_dict in for_charts.items():
        for model_name, queries in model_dict.items():
            for q in queries:
                setattr(q.parameters, 'gruppo', 'noto' if test_name in noti else 'ignoto')
    #rinominiamo i test con nomi brevi
    rinomine = {'cities/global_liveability':'GLI', 'cities/city_free_score':'Average', 'planets/dif':'DIF', 'planets/esi':'ESI', 'purchases/customer_segmentation':'RFM', 'purchases/customer_cbs':'TVA', 'molecules/levenshtein':'LEV','molecules/RWED':'RWED'}
    keys = tuple(for_charts.keys())
    for test_name in keys:
        if test_name in rinomine:
            for_charts[rinomine[test_name]] = for_charts.pop(test_name)
    #trasforma deepseek-V3-togheter -> deepseek-V3
    ds_name = 'deepseek-V3-together'
    for test_name, model_dict in for_charts.items():
        if ds_name in model_dict:
            model_dict['deepseek-V3'] = model_dict.pop(ds_name)
    # trasforma k in percentuale
    for test_name, model_dict in for_charts.items():
        for model_name, queries in model_dict.items():
            for q in queries:
                if q.parameters.k is not None and q.parameters.n_elems is not None:
                    pre_round = q.parameters.k / q.parameters.n_elems
                    q.parameters.k = _round_scalar(pre_round)
    #transforma mare in percentuale inversa
    for test_name, model_dict in for_charts.items():
        for model_name, queries in model_dict.items():
            for q in queries:
                try:
                    q.evaluations.mare = 1 - q.evaluations.mare / len(q.ground_truth)
                    q.evaluations.mare_k = 1 - q.evaluations.mare_k / len(q.ground_truth)
                except (AttributeError, KeyError):
                    pass
    return for_charts

def prepare_for_dashboard() -> dict[str, dict[str, dict[str, list[Query]]]]:
    runs = load_experiments()
    tot_tests = 0
    executed_tests = 0
    result = {}
    for run in runs:
        for_dashboard: dict[str, dict[str, dict[str, list[Query]]]] = {}
        for query in run.queries:
            for_dashboard[query.name] = {}
            for run_type_experiment in query.run_type_experiments:
                for_dashboard[query.name][run_type_experiment.run_type.name] = {}
                for e in run_type_experiment.experiments:
                    tot_tests += 1
                    path = e.inner_folder / 'evaluated_queries.pkl'
                    if path.exists():
                        executed_tests += 1
                        e.test.pickle_to_queries(path)
                        for_dashboard[query.name][run_type_experiment.run_type.name][e.model.name_path] = e.test.queries

        result[run.name_path] = prepare_for_charts(for_dashboard)

    print(f"showing {executed_tests}/{tot_tests} tests")
    return result


if __name__ == "__main__":
    runs: list[Run] = load_experiments()
    if len(runs) == 0:
        print("No (valid) experiments found.")
        exit(1)

    for run in runs: # per ogni run folder
        print(f"*** Running experiments for run: {run.name_path} ***\n")
        for query in run.queries:
            print(f"=== Test: {query.name} ===\n")
            for run_type_experiment in query.run_type_experiments:
                print(f"--- Run type: {run_type_experiment.run_type} ---\n")
                experiments = run_type_experiment.experiments
                for experiment in experiments:
                    print(f"Running experiment: {experiment.name}\n")
                    experiment.execute()
                    print('\n')

                if run.create_aggregated_csv:
                    run_type_experiment.all_evaluated = all(e.test.queries and all(q.evaluations is not None
                                                                                    for q in e.test.queries)
                                                            for e in experiments)
                    if run_type_experiment.all_evaluated:
                        if len(experiments) <= 1 or len(set([e.test.run_type for e in experiments])) != 1:
                            print("Not enough evaluated experiments with the same run type to aggregate results, skipping aggregation.")
                        else:
                            print(f"aggregating results for test: {query.name} - {run_type_experiment.run_type}")
                            merged_df = merge_queries([e.test.queries for e in experiments], [e.model.name_path for e in experiments])
                            merged_df.to_csv(experiments[0].test.run_folder.parent / 'results' / 'aggregated_queries.csv', index=False, decimal=',', sep=';')
                    else:
                        print(f"Some experiments for test {query.name} were not evaluated, skipping aggregation")
                print('\n')