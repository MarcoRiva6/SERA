from experimenter import get_expr_filenames_from_args, Run, parse_expr_file, expr_folder

if __name__ == "__main__":
    experiment_files = get_expr_filenames_from_args()
    if len(experiment_files) != 1:
        raise ValueError("Multiple experiment file execution is not supported")
    run: Run = parse_expr_file(expr_folder / experiment_files[0])

    for query in run.queries:
        print(f"=== Test: {query.name} ===\n")
        for run_type_experiment in query.run_type_experiments:
            print(f"--- Run type: {run_type_experiment.run_type} ---\n")
            experiments = run_type_experiment.experiments
            experiment = experiments[0]
            experiment.sample()