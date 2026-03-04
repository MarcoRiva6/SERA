from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ALL


# ----------------------------
# Introspection helpers
# ----------------------------

def safe_sort_val(v: Any) -> Tuple[int, float, str]:
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    try:
        return (1, float(v), "")
    except (ValueError, TypeError):
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
        return [outer, inner]
    return [" | ".join("" if v is None else str(v) for v in k) for k in keys]


# ----------------------------
# Core: per-set “schema” extraction
# ----------------------------

def compute_set_schema(datasets: Dict[str, List[Any]]):
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

    if not dataset_sets:
        raise ValueError("dataset_sets is empty")

    set_names = list(dataset_sets.keys())
    default_set = set_names[0]

    schemas = {}
    for set_name, datasets in dataset_sets.items():
        schemas[set_name] = compute_set_schema(datasets)

    preferred_order = ["ndcg_scores", "ndcg_k", "mare", "mare_k", "kendall", "kendall_k", "spearman", "spearman_k"]
    available_metrics = {m for (ds_names, pf, ef, vbp) in schemas.values() for m in ef if m != "hallucination_rate"}

    all_eval_fields = []
    for metric in preferred_order:
        if metric in available_metrics:
            all_eval_fields.append(metric)
            available_metrics.remove(metric)

    all_eval_fields.extend(sorted(available_metrics))
    if not all_eval_fields:
        raise ValueError("No evaluation fields found in any set.")

    # Aggiunto suppress_callback_exceptions=True per gestire i filtri dinamici senza warning
    app = Dash(__name__, suppress_callback_exceptions=True)

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
                    html.Div(
                        id="filters-container",
                        style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "flex-end"},
                    ),
                ],
            ),

            # NUOVI TABS PER SELEZIONARE IL TIPO DI GRAFICO
            dcc.Tabs(
                id="chart-type-tabs",
                value="line",
                children=[
                    dcc.Tab(label="Line Plots (Medie)", value="line"),
                    dcc.Tab(label="Box Plots (Distribuzioni)", value="box"),
                ],
                style={"marginTop": "20px", "marginBottom": "5px"}
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

    # --- Update controls (questo era il pezzo sparito!) ---
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
        ds_value = dataset_names[:]

        x_options = [{"label": f, "value": f} for f in param_fields]
        x_value = param_fields[:2] if len(param_fields) >= 2 else param_fields[:1]

        filters_ui = []
        for f in param_fields:
            filters_ui.append(
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Div(f"Filter: {f}", style={"fontWeight": 700, "marginBottom": "6px"}),
                        dcc.Dropdown(
                            id={"type": "filter", "field": f},
                            options=[{"label": str(v), "value": v} for v in values_by_param[f]],
                            value=[],
                            multi=True,
                            placeholder="All",
                        ),
                    ],
                )
            )

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

    # --- Update all figures (Line e Box unificati) ---
    @app.callback(
        Output({"type": "metric-graph", "metric": all_eval_fields[0]}, "figure"),
        *[Output({"type": "metric-graph", "metric": m}, "figure") for m in all_eval_fields[1:]],
        Input("set-tabs", "value"),
        Input("chart-type-tabs", "value"),
        Input("datasets-select", "value"),
        Input("x-fields-select", "value"),
        Input({"type": "filter", "field": ALL}, "value"),
        State({"type": "filter", "field": ALL}, "id"),
    )
    def update_figures(active_set: str, chart_type: str, selected_datasets, x_fields, filter_values, filter_ids):
        datasets = dataset_sets[active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_set]

        selected_datasets = selected_datasets or dataset_names
        x_fields = (x_fields or param_fields[:1])[:]

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

        filtered_by_ds: Dict[str, List[Any]] = {}
        for ds in selected_datasets:
            # 1. Filtra le query
            _qs = [q for q in datasets[ds] if matches_filters(q)]

            # 2. Ordinale in base ai valori selezionati sull'asse X
            def q_sort_key(q):
                return tuple(safe_sort_val(get_param_value(q, f)) for f in x_fields)

            # 3. Salva la lista già perfettamente ordinata
            filtered_by_ds[ds] = sorted(_qs, key=q_sort_key)

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

            for ds in selected_datasets:
                qs = filtered_by_ds[ds]

                if chart_type == "line":
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

                elif chart_type == "box":
                    ds_y = []
                    ds_x_outer = []
                    ds_x_inner = []
                    ds_x_single = []

                    for q in qs:
                        v = get_eval_value(q, metric_name)
                        if isinstance(v, (int, float)):
                            key = tuple(get_param_value(q, f) for f in x_fields)
                            ds_y.append(float(v))

                            if len(x_fields) == 2:
                                ds_x_outer.append("" if key[0] is None else str(key[0]))
                                ds_x_inner.append("" if key[1] is None else str(key[1]))
                            elif len(x_fields) == 1:
                                ds_x_single.append("" if key[0] is None else str(key[0]))
                            else:
                                ds_x_single.append("all")

                    if not ds_y:
                        continue

                    if len(x_fields) == 2:
                        x_plot_box = [ds_x_outer, ds_x_inner]
                    else:
                        x_plot_box = ds_x_single

                    fig.add_trace(
                        go.Box(
                            x=x_plot_box,
                            y=ds_y,
                            name=ds,
                            boxpoints="outliers",
                        )
                    )

            layout_kwargs = {
                "title": {"text": f"<b>{metric_name}</b>", "x": 0.5, "y": 0.97, "xanchor": "center", "font": {"size": 16}},
                "margin": {"l": 50, "r": 20, "t": 90, "b": 90},
                "height": 420,
                "legend": {"orientation": "h", "y": -0.25},
            }

            if chart_type == "box":
                layout_kwargs["boxmode"] = "group"

            fig.update_layout(**layout_kwargs)

            if chart_type == "box":
                layout_kwargs["boxmode"] = "group"

            fig.update_layout(**layout_kwargs)

            # --- IMPOSTAZIONE ASSI RIPRISTINATA ---
            if len(x_fields) == 2:
                fig.update_xaxes(type="multicategory", categoryorder="trace")
            else:
                fig.update_xaxes(type="category", categoryorder="trace")

            fig.update_yaxes(rangemode="tozero")

            figures.append(fig)

        return tuple(figures)

    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    raise SystemExit("Replace this with your dataset_sets dict and call run_multi_set_dashboard(dataset_sets).")