from dataclasses import dataclass
from pathlib import Path

from models.model import Model, SubmissionError
from queries.test import Test

from experiments.run_type import RunType

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def interactive_plot(df: pd.DataFrame, x: str, y: str, kind: str = "line"):
    if kind == "line":
        fig = px.line(df, x=x, y=y)
    elif kind == "scatter":
        fig = px.scatter(df, x=x, y=y)
    elif kind == "bar":
        fig = px.bar(df, x=x, y=y)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    fig.show()
    return fig

def more_interactive_plot(df: pd.DataFrame, title: str, columns: list[str] = None, filters: list[str] = []):
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    initial_y = columns[0]

    # Create initial plot: x = row numbers
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df.index,      # row numbers
                y=df[initial_y], # default y column
                mode="markers"
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Row Number",
        yaxis_title=initial_y,
    )

    # ---- Dropdown buttons for Y-axis ----
    if len(columns) > 1:
        y_buttons = []
        for col in columns:
            y_buttons.append(
                dict(
                    label=col,
                    method="update",
                    args=[
                        {"y": [df[col]]},             # update y values
                        {"yaxis": {"title": col}},    # update y label
                    ],
                )
            )

        # ---- Put dropdown above chart ----
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=y_buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ],
            margin=dict(t=80)
        )
    if len(filters) > 0:
        for i, f in enumerate(filters):
            filter_buttons = []
            filter_buttons.append(
                dict(
                    label="All",
                    method="update",
                    args=[
                        {"x": [df.index], "y": [df[initial_y]]},
                        {"xaxis": {"title": "Row Number"}, "yaxis": {"title": initial_y}},
                    ],
                )
            )
            unique_values = df[f].unique()
            for val in unique_values:
                filtered_df = df[df[f] == val].reset_index(drop=True)
                filter_buttons.append(
                    dict(
                        label=str(val),
                        method="update",
                        args=[
                            {"x": [filtered_df.index], "y": [filtered_df[initial_y]]},
                            {"xaxis": {"title": "Row Number"}, "yaxis": {"title": initial_y}},
                        ],
                    )
                )
            # Add filter dropdown to layout
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=filter_buttons,
                        direction="down",
                        showactive=True,
                        x=1.0 + i * 0.2,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                    )
                ],
                margin=dict(t=80)
            )

    fig.show()

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

    def __post_init__(self):
        self.inner_folder = self.run_folder / self.test.family / self.test.name_path / 'results' / self.model.name_path / self.run_type

    def display(self, df: pd.DataFrame = None) -> None:
        if df is None:
            df = pd.read_parquet(self.inner_folder / 'queries.parquet')
        more_interactive_plot(df=df, title=self.name, columns=['ndcg_k'], filters=['prompt_level'])

    def execute(self) -> None:
        """
        Execute the experiment by preparing the test, submitting queries to the model and saving results.
        """
        test_has_already_run_once = (self.test.run_folder / self.test.params_file_name).exists()
        if test_has_already_run_once and not self.test.same_params():
            print('Test parameters have changed since last execution. Stopping to avoid inconsistencies')
            return

        print('Preparing test...')
        self.test.save_params()
        queries_file_exists: bool = (self.test.run_folder / self.test.prepared_queries_file_name).exists()
        if queries_file_exists:
            print('Restoring already generated queries...')
            self.test.restore_queries()
        else:
            print('Generating queries...')
            self.test.generate_queries()
            print('Storing queries...')
            self.test.store_queries()
            self.test.queries_to_csv('prepared_queries.csv')
        print('test ready.')
        try:
            self.model.run(self.test.queries)
            print('test submission complete.')
        except NotImplementedError as e:
            print(e)
            return
        except SubmissionError as e:
            print('Submission error:', e)
            return
        except Exception as e:
            print('An unknown error occurred during submission:', e)
            return

        at_least_one_completed = any(q.response is not None for q in self.test.queries)
        if at_least_one_completed:
            print('Evaluating queries...')
            for q in self.test.queries:
                if q.response is not None:
                    q.evaluations = self.test.evaluate_query(q)

            print('Storing answered queries...')
            self.test.queries_to_csv('answered_queries.csv', self.inner_folder)
            print('queries saved to csv.')
            self.test.evaluate()
            print('test evaluation:', self.test.evaluations)
            self.test.evaluations_to_csv(dest=self.inner_folder)
            print('evaluations saved to csv.')

        #self.display(df=q_df)

        print('test execution completed.')
