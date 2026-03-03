from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ALL


# ----------------------------
# Introspection helpers
# ----------------------------

def safe_sort_val(v: Any) -> Tuple[int, float, str]:
    """
    Ordina in modo sicuro: prima i None, poi i numeri (o stringhe numeriche),
    infine le stringhe normali.
    """
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    try:
        # Cerca di convertire in numero se è una stringa numerica (es. "11")
        return (1, float(v), "")
    except (ValueError, TypeError):
        # Fallback per le stringhe di puro testo
        return (2, 0.0, str(v))


def dataclass_field_names(dc_type: type) -> List[str]:
    return [f.name for f in fields(dc_type)]

def get_param_fields_from_queries(queries: List[Any]) -> List[str]:
    if not queries:
        return []
    params = queries[0].parameters
    if not is_dataclass(params):
        raise TypeError("Query.parameters must be a dataclass instance.")
    return dataclass_field_names(type(params))

def get_eval_fields_from_queries(queries: List[Any]) -> List[str]:
    if not queries:
        return []
    ev = queries[0].evaluations
    if not is_dataclass(ev):
        raise TypeError("Query.evaluations must be a dataclass instance.")
    return dataclass_field_names(type(ev))

def get_param_value(q: Any, field_name: str) -> Any:
    return getattr(q.parameters, field_name, None)

def get_eval_value(q: Any, metric_name: str) -> Any:
    return getattr(q.evaluations, metric_name, None)

def mean(values: List[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


# ----------------------------
# Multicategory axis helpers
# ----------------------------

def build_category_keys(queries: List[Any], x_fields: List[str]) -> List[Tuple[Any, ...]]:
    keys = set()
    for q in queries:
        keys.add(tuple(get_param_value(q, f) for f in x_fields))

    def sort_key(t: Tuple[Any, ...]):
        return tuple(safe_sort_val(v) for v in t)

    return sorted(keys, key=sort_key)

def plotly_x_from_keys(keys: List[Tuple[Any, ...]], x_fields: List[str]):
    if len(x_fields) == 0:
        return ["all"]
    if len(x_fields) == 1:
        return [("" if k[0] is None else str(k[0])) for k in keys]
    if len(x_fields) == 2:
        outer = [("" if k[0] is None else str(k[0])) for k in keys]
        inner = [("" if k[1] is None else str(k[1])) for k in keys]
        return [outer, inner]  # true multicategory
    return [" | ".join("" if v is None else str(v) for v in k) for k in keys]


# ----------------------------
# Core: per-set “schema” extraction
# ----------------------------

def compute_set_schema(datasets: Dict[str, List[Any]]):
    """
    From datasets (model -> list[Query]), infer:
    - dataset_names
    - param_fields
    - eval_fields
    - values_by_param (unique values for each param field)
    """
    if not datasets:
        raise ValueError("Empty datasets in set.")

    dataset_names = list(datasets.keys())
    first_ds = dataset_names[0]
    first_queries = datasets[first_ds]
    if not first_queries:
        raise ValueError(f"Dataset '{first_ds}' is empty.")

    param_fields = get_param_fields_from_queries(first_queries)
    eval_fields = get_eval_fields_from_queries(first_queries)

    values_by_param: Dict[str, List[Any]] = {}
    for f in param_fields:
        vals = set()
        for qs in datasets.values():
            for q in qs:
                vals.add(get_param_value(q, f))
        # <-- RIGA MODIFICATA:
        values_by_param[f] = sorted(list(vals), key=safe_sort_val)

    return dataset_names, param_fields, eval_fields, values_by_param


# ----------------------------
# Main: multi-set dashboard
# ----------------------------

def run_multi_set_dashboard(
        dataset_sets: Dict[str, Dict[str, List[Any]]],
        *,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = True,
):
    """
    dataset_sets:
      {
        "Set A": { "model1": [Query...], "model2": [Query...] },
        "Set B": { "model1": [Query...], "model2": [Query...] },
      }
    """

    if not dataset_sets:
        raise ValueError("dataset_sets is empty")

    set_names = list(dataset_sets.keys())
    default_set = set_names[0]

    # Precompute schema per set (fast + avoids recomputing on every callback)
    schemas = {}
    for set_name, datasets in dataset_sets.items():
        schemas[set_name] = compute_set_schema(datasets)

    # Ordine desiderato delle metriche
    preferred_order = ["ndcg_scores", "ndcg_k", "mare", "mare_k", "kendall", "kendall_k", "spearman", "spearman_k"]

    # Raccogli tutte le metriche disponibili (escludendo hallucination_rate se necessario)
    available_metrics = {m for (ds_names, pf, ef, vbp) in schemas.values() for m in ef if m != "hallucination_rate"}

    # Costruisci la lista finale: prima quelle nell'ordine preferito, se esistono
    all_eval_fields = []
    for metric in preferred_order:
        if metric in available_metrics:
            all_eval_fields.append(metric)
            available_metrics.remove(metric) # Rimuovi per non duplicare

    # Poi aggiungi tutte le altre metriche rimanenti, in ordine alfabetico
    all_eval_fields.extend(sorted(available_metrics))
    if not all_eval_fields:
        raise ValueError("No evaluation fields found in any set.")

    app = Dash(__name__)

    # --- Top bar controls (datasets + x-fields + filters) ---
    # Filters are created dynamically, but IDs remain stable via pattern-matching.
    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "12px"},
        children=[
            html.H2("Queries Dashboard", style={"margin": "0 0 10px 0"}),

            dcc.Tabs(
                id="set-tabs",
                value=default_set,
                children=[dcc.Tab(label=name, value=name) for name in set_names],
            ),

            html.Div(
                id="controls-bar",
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "14px",
                    "alignItems": "flex-end",
                    "border": "1px solid #ddd",
                    "borderRadius": "10px",
                    "padding": "12px",
                    "marginTop": "10px",
                },
                children=[
                    html.Div(
                        style={"minWidth": "320px"},
                        children=[
                            html.Div("Datasets (lines)", style={"fontWeight": 700, "marginBottom": "6px"}),
                            dcc.Dropdown(id="datasets-select", multi=True, clearable=False),
                        ],
                    ),
                    html.Div(
                        style={"minWidth": "420px"},
                        children=[
                            html.Div("X-axis categories (choose 1–2)", style={"fontWeight": 700, "marginBottom": "6px"}),
                            dcc.Dropdown(id="x-fields-select", multi=True, clearable=False),
                        ],
                    ),

                    # Filters container (dynamic)
                    html.Div(
                        id="filters-container",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "flex-end"},
                    ),
                ],
            ),

            html.Div(
                id="charts-grid",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "14px",
                    "marginTop": "14px",
                },
                children=[
                    dcc.Graph(
                        id={"type": "metric-graph", "metric": m},
                        config={"displayModeBar": True},
                        style={"height": "420px"},
                    )
                    for m in all_eval_fields
                ],
            ),
        ],
    )

    # --- Update controls (dataset dropdown, x-field dropdown, filters) when tab changes ---
    @app.callback(
        Output("datasets-select", "options"),
        Output("datasets-select", "value"),
        Output("x-fields-select", "options"),
        Output("x-fields-select", "value"),
        Output("filters-container", "children"),
        Input("set-tabs", "value"),
    )
    def update_controls(active_set: str):
        datasets = dataset_sets[active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_set]

        ds_options = [{"label": d, "value": d} for d in dataset_names]
        ds_value = dataset_names[:]  # default: all

        x_options = [{"label": f, "value": f} for f in param_fields]
        x_value = param_fields[:2] if len(param_fields) >= 2 else param_fields[:1]

        # Build multi-select filters (empty = no filter)
        filters_ui = []
        for f in param_fields:
            filters_ui.append(
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Div(f"Filter: {f}", style={"fontWeight": 700, "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id={"type": "filter", "field": f},
                            options=[{"label": str(v), "value": v} for v in sorted(values_by_param[f])],
                            value=[],
                            multi=True,
                            placeholder="All",
                        ),
                    ],
                )
            )

        # Add special filter for parsing_failed if it exists
        filters_ui.append(
            html.Div(
                style={"minWidth": "220px"},
                children=[
                    html.Div("Filter: parsing_failed", style={"fontWeight": 700, "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id={"type": "filter", "field": "parsing_failed"},
                        options=[
                            {"label": "True", "value": True},
                            {"label": "False", "value": False}
                        ],
                        value=[],
                        multi=True,
                        placeholder="All",
                    ),
                ],
            )
        )

        return ds_options, ds_value, x_options, x_value, filters_ui

    # --- Update all figures ---
    @app.callback(
        Output({"type": "metric-graph", "metric": all_eval_fields[0]}, "figure"),
        *[Output({"type": "metric-graph", "metric": m}, "figure") for m in all_eval_fields[1:]],
        Input("set-tabs", "value"),
        Input("datasets-select", "value"),
        Input("x-fields-select", "value"),
        Input({"type": "filter", "field": ALL}, "value"),
        State({"type": "filter", "field": ALL}, "id"),
    )
    def update_figures(active_set: str, selected_datasets, x_fields, filter_values, filter_ids):
        datasets = dataset_sets[active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_set]

        selected_datasets = selected_datasets or dataset_names
        x_fields = (x_fields or param_fields[:1])[:]

        # Build filters dict: field -> list of allowed values (empty => no filter)
        filters: Dict[str, List[Any]] = {}
        for fid, vals in zip(filter_ids, filter_values):
            filters[fid["field"]] = vals or []

        def matches_filters(q: Any) -> bool:
            for f, allowed in filters.items():
                if not allowed:
                    continue

                val = getattr(q, "parsing_failed", None) if f == "parsing_failed" else get_param_value(q, f)
                if val not in allowed:
                    return False
            return True

        # Filter queries by set+filters first
        filtered_by_ds: Dict[str, List[Any]] = {}
        for ds in selected_datasets:
            filtered_by_ds[ds] = [q for q in datasets[ds] if matches_filters(q)]

        # Build global x categories from filtered data
        all_keys_set = set()
        for ds in selected_datasets:
            all_keys_set.update(build_category_keys(filtered_by_ds[ds], x_fields=x_fields))

        def sort_key(t: Tuple[Any, ...]):
            return tuple(safe_sort_val(v) for v in t)

        all_keys = sorted(all_keys_set, key=sort_key)
        x_plotly = plotly_x_from_keys(all_keys, x_fields)

        figures = []

        for metric_name in all_eval_fields:
            fig = go.Figure()

            # One line per dataset
            for ds in selected_datasets:
                qs = filtered_by_ds[ds]

                y_map: Dict[Tuple[Any, ...], float | None] = {}
                for key in all_keys:
                    vals = []
                    for q in qs:
                        if tuple(get_param_value(q, f) for f in x_fields) != key:
                            continue
                        v = get_eval_value(q, metric_name)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                    y_map[key] = mean(vals)

                y_aligned = [y_map.get(k) for k in all_keys]

                fig.add_trace(
                    go.Scatter(
                        x=x_plotly,
                        y=y_aligned,
                        mode="lines+markers",
                        name=ds,
                        connectgaps=False,
                    )
                )

            # Bold title + safe margins
            fig.update_layout(
                title={
                    "text": f"<b>{metric_name}</b>",
                    "x": 0.5,
                    "y": 0.97,
                    "xanchor": "center",
                    "font": {"size": 16},
                },
                margin={"l": 50, "r": 20, "t": 90, "b": 90},
                height=420,
                legend={"orientation": "h", "y": -0.25},
            )

            # True multicategory when 2 fields
            if len(x_fields) == 2:
                fig.update_xaxes(type="multicategory")
            else:
                fig.update_xaxes(type="category")

            fig.update_yaxes(rangemode="tozero")

            figures.append(fig)

        return tuple(figures)

    # IMPORTANT: do not restart your main process
    app.run(host=host, port=port, debug=debug, use_reloader=False)


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Build your multi-set structure like this:
    #
    # dataset_sets = {
    #   "Set 1": {
    #       "deepseek-v3": deepseek_v3_queries,
    #       "gemini-pro": gemini_pro_queries,
    #   },
    #   "Set 2": {
    #       "deepseek-v3": deepseek_v3_queries_other_run,
    #       "gemini-pro": gemini_pro_queries_other_run,
    #   },
    # }
    #
    # run_multi_set_dashboard(dataset_sets)

    raise SystemExit("Replace this with your dataset_sets dict and call run_multi_set_dashboard(dataset_sets).")