from __future__ import annotations

import copy
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State, ALL

from queries.test import Query
import plotly.colors


# ----------------------------
# Introspection helpers
# ----------------------------

font = 16 #26 #30
tickfont = font-2

def safe_sort_val(v: Any) -> Tuple[int, float, str]:
    if v is None:
        return (0, 0.0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    try:
        # Prova a convertire in numero (se è una stringa numerica)
        return (1, float(v), "")
    except (ValueError, TypeError):
        # Se è una stringa vera e propria, gestiamo i casi custom
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower == "easy":
                return (2, 0.0, "01_easy")
            elif v_lower == "medium":
                return (2, 0.0, "02_medium")
            elif v_lower == "hard":
                return (2, 0.0, "03_hard")

        # Per tutte le altre stringhe, usa l'ordinamento alfabetico normale
        return (2, 0.0, str(v))

def dataclass_field_names(dc_type: type) -> List[str]:
    return [f.name for f in fields(dc_type)]

def get_param_fields_from_queries(queries: List[Any]) -> List[str]:
    if not queries:
        return []
    params = queries[0].parameters
    if not is_dataclass(params):
        raise TypeError("Query.parameters must be a dataclass instance.")

    fields_list = dataclass_field_names(type(params))

    # Supporto per l'attributo iniettato per il tab "All Experiments"
    if hasattr(params, "expr") and "expr" not in fields_list:
        fields_list.append("expr")
    # supporto per raggruppamento esperimenti
    if hasattr(params, "gruppo") and "gruppo" not in fields_list:
        fields_list.append("gruppo")
    #support per raggruppamento dataset
    if hasattr(params, "dataset") and "dataset" not in fields_list:
        fields_list.append("dataset")
    # supporto per run_type
    if hasattr(params, "run_type") and "run_type" not in fields_list:
        fields_list.append("run_type")
    # supporto per raggruppamento difficoltà
    if hasattr(params, "difficulty") and "difficulty" not in fields_list:
        fields_list.append("difficulty")
    # supporto a partizionabile
    if hasattr(params, "partizionabile") and "partizionabile" not in fields_list:
        fields_list.append("partizionabile")
    # supporto ad anonimizzabile
    if hasattr(params, "anonimizzabile") and "anonimizzabile" not in fields_list:
        fields_list.append("anonimizzabile")
    # supporto ad anonimizzabile
    if hasattr(params, "generalizzabile") and "generalizzabile" not in fields_list:
        fields_list.append("generalizzabile")

    return fields_list

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
# Main: MULTI-SUITE DASHBOARD
# ----------------------------

def run_dashboard(
        experiment_suites: Dict[str, Dict[str, Dict[str, List[Query]]]],
        *,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = True,
):
    if not experiment_suites:
        raise ValueError("experiment_suites is empty")

        # --- CONFIGURAZIONE EXPORT GLOBALE ---
        # Modificato in formato PNG ad alta risoluzione (scale=3)
    def get_global_config(metric_name):
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',                    # <--- Esporta in formato PNG
                'filename': f'export_{metric_name}',
                'scale': 3#,
                #'width': 1000,   # <--- BLOCCA LA LARGHEZZA BASE (es. 800 o 900)
                #'height': 450   # <--- BLOCCA L'ALTEZZA (uguale al tuo layout) # <--- IMPORTANTE per la risoluzione
            }
        }

        # --- PRE-PROCESSING DELLE SUITE (Invariato) ---
    processed_suites = {}
    schemas = {}

    for suite_name, dataset_sets in experiment_suites.items():
        global_datasets = {}
        for set_name, datasets in dataset_sets.items():
            for ds_name, queries in datasets.items():
                if ds_name not in global_datasets:
                    global_datasets[ds_name] = []
                for original_q in queries:
                    q = copy.deepcopy(original_q)
                    setattr(q.parameters, "expr", set_name)
                    global_datasets[ds_name].append(q)

        new_dataset_sets = {"All Experiments": global_datasets}
        new_dataset_sets.update(dataset_sets)
        processed_suites[suite_name] = new_dataset_sets

        schemas[suite_name] = {}
        for set_name, datasets in new_dataset_sets.items():
            schemas[suite_name][set_name] = compute_set_schema(datasets)

    # Impostazioni di default iniziali
    suite_names = list(processed_suites.keys())
    default_suite = suite_names[0]
    default_set = list(processed_suites[default_suite].keys())[0]

    # Trova tutte le metriche globalmente
    preferred_order = ["ndcg_scores", "ndcg_k", "mare", "mare_k", "kendall", "kendall_k", "spearman", "spearman_k"]
    available_metrics = {m for suite in schemas.values() for (ds_names, pf, ef, vbp) in suite.values() for m in ef}

    all_eval_fields = []
    for metric in preferred_order:
        if metric in available_metrics:
            all_eval_fields.append(metric)
            available_metrics.remove(metric)

    all_eval_fields.extend(sorted(available_metrics))
    if not all_eval_fields:
        raise ValueError("No evaluation fields found in any suite.")

    # Mappa colori globale fissa (codici HEX estratti esattamente dall'immagine di riferimento)
    GLOBAL_COLOR_MAP = {
        "deepseek": "#EF553B",               # Rosso/Arancio
        "deepseek-thinking": "#00CC96",      # Verde acqua
        "gemini": "#AB63FA",          # Viola
        "gemini-thinking": "#FFA15A", # Arancio/Giallo chiaro
        "gemma3-27b": "#19D3F3",                # Azzurro
        "qwen3-30b": "#FF6692",                 # Rosa acceso/Rosso
        "gemma3-12b": "#B6E880",                # Verde lime chiaro
        "qwen3-14b": "#FF97FF"                  # Rosa chiaro
    }

    app = Dash(__name__, suppress_callback_exceptions=True)

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "12px", "backgroundColor": "#f4f6f9", "minHeight": "100vh"},
        children=[
            html.H2("Queries Dashboard", style={"margin": "0 0 10px 0"}),

            # LIVELLO 1: I MACRO-TAB (Le Suite di esperimenti)
            dcc.Tabs(
                id="suite-tabs",
                value=default_suite,
                children=[dcc.Tab(label=name, value=name) for name in suite_names],
                style={"marginBottom": "10px", "fontWeight": "bold"}
            ),

            # LIVELLO 2: I TAB DEGLI ESPERIMENTI
            dcc.Tabs(
                id="set-tabs",
                value=default_set,
                children=[dcc.Tab(label=name, value=name) for name in processed_suites[default_suite].keys()],
            ),

            html.Div(
                id="controls-bar",
                style={
                    "display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "flex-end",
                    "border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "10px",
                    "backgroundColor": "white"
                },
                children=[
                    html.Div(style={"minWidth": "320px"}, children=[
                        html.Div("Datasets (lines/cards)", style={"fontWeight": 700, "marginBottom": "6px"}),
                        dcc.Dropdown(id="datasets-select", multi=True, clearable=False),
                    ]),
                    html.Div(style={"minWidth": "420px"}, children=[
                        html.Div("X-axis categories (choose 1–2)", style={"fontWeight": 700, "marginBottom": "6px"}),
                        dcc.Dropdown(id="x-fields-select", multi=True, clearable=False),
                    ]),
                    html.Div(id="filters-container", style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "flex-end"}),
                ],
            ),

            # LIVELLO 3: I TAB DEI GRAFICI / RIEPILOGO / PARETO
            dcc.Tabs(
                id="chart-type-tabs",
                value="summary",
                children=[
                    dcc.Tab(label="Riepilogo (Statistiche)", value="summary"),
                    dcc.Tab(label="Scatter Plots (Medie)", value="line"), # (L'avevamo rinominato in Scatter Plots)
                    dcc.Tab(label="Box Plots (Distribuzioni)", value="box"),
                    dcc.Tab(label="Heatmap (Correlazioni)", value="heatmap"),
                    dcc.Tab(label="Bar Plot (Correlazione Binaria)", value="bar_corr"), # <--- NUOVO TAB
                    dcc.Tab(label="Fronte di Pareto", value="pareto"),
                ],
                style={"marginTop": "20px", "marginBottom": "5px"}
            ),

            html.Div(id="summary-container", style={"display": "block", "marginTop": "14px"}),

            # ---> ECCO IL PEZZO CHE CAUSA L'ERRORE SE MANCA! <---
            html.Div(id="pareto-container", style={"display": "none", "marginTop": "14px"}),

            # --- APPLICAZIONE DELLA CONFIGURAZIONE AI GRAFICI ---
            html.Div(
                id="charts-grid",
                style={"display": "none", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "14px", "marginTop": "14px"},
                children=[
                    dcc.Graph(
                        id={"type": "metric-graph", "metric": m},
                        config=get_global_config(m), # <--- APPLICATO QUI
                        style={"height": "450px"}
                    )
                    for m in all_eval_fields
                ],
            ),
        ],
    )

    # --- CALLBACK 0: Mostra/Nascondi vista Riepilogo, Pareto o Grafici ---
    @app.callback(
        Output("summary-container", "style"),
        Output("pareto-container", "style"),
        Output("charts-grid", "style"),
        Input("chart-type-tabs", "value")
    )
    def toggle_views(chart_type):
        if chart_type == "summary":
            return {"display": "block", "marginTop": "14px"}, {"display": "none"}, {"display": "none"}
        elif chart_type == "pareto":
            return {"display": "none"}, {"display": "block", "marginTop": "14px"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "none"}, {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "14px", "marginTop": "14px"}

    # --- CALLBACK 1: Aggiorna i tab degli esperimenti quando cambi la Suite ---
    @app.callback(
        Output("set-tabs", "children"),
        Output("set-tabs", "value"),
        Input("suite-tabs", "value")
    )
    def update_set_tabs(active_suite):
        suite_data = processed_suites[active_suite]
        tabs = [dcc.Tab(label=name, value=name) for name in suite_data.keys()]
        return tabs, list(suite_data.keys())[0]

    # --- CALLBACK 2: Aggiorna i controlli in base a Suite ed Esperimento ---
    @app.callback(
        Output("datasets-select", "options"),
        Output("datasets-select", "value"),
        Output("x-fields-select", "options"),
        Output("x-fields-select", "value"),
        Output("filters-container", "children"),
        Input("set-tabs", "value"),
        Input("suite-tabs", "value"),
    )
    def update_controls(active_set: str, active_suite: str):
        if not active_set or active_set not in processed_suites.get(active_suite, {}):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_suite][active_set]

        ds_options = [{"label": d, "value": d} for d in dataset_names]
        ds_value = dataset_names[:]

        x_options = [{"label": f, "value": f} for f in param_fields]
        x_value = param_fields[:2] if len(param_fields) >= 2 else param_fields[:1]

        filters_ui = []
        for f in param_fields:
            filters_ui.append(html.Div(style={"minWidth": "220px"}, children=[
                html.Div(f"Filter: {f}", style={"fontWeight": 700, "marginBottom": "6px"}),
                dcc.Dropdown(
                    id={"type": "filter", "field": f},
                    options=[{"label": str(v), "value": v} for v in values_by_param[f]],
                    value=[], multi=True, placeholder="All"
                ),
            ]))

        filters_ui.append(html.Div(style={"minWidth": "220px"}, children=[
            html.Div("Filter: parsing_failed", style={"fontWeight": 700, "marginBottom": "6px"}),
            dcc.Dropdown(
                id={"type": "filter", "field": "parsing_failed"},
                options=[{"label": "True", "value": True}, {"label": "False", "value": False}],
                value=[], multi=True, placeholder="All"
            ),
        ]))

        return ds_options, ds_value, x_options, x_value, filters_ui

    # --- CALLBACK 3: Popola il pannello di Riepilogo (Summary) ---
    @app.callback(
        Output("summary-container", "children"),
        Input("set-tabs", "value"),
        Input("datasets-select", "value"),
        Input({"type": "filter", "field": ALL}, "value"),
        Input("suite-tabs", "value"),
        State({"type": "filter", "field": ALL}, "id"),
    )
    def update_summary(active_set, selected_datasets, filter_values, active_suite, filter_ids):
        if not active_set or active_set not in processed_suites.get(active_suite, {}):
            return dash.no_update

        datasets = processed_suites[active_suite][active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_suite][active_set]
        selected_datasets = selected_datasets or dataset_names

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

        cards = []
        for ds in selected_datasets:
            qs = [q for q in datasets[ds] if matches_filters(q)]
            total = len(qs)
            failed = sum(1 for q in qs if getattr(q, "parsing_failed", False) is True)
            completed = sum(1 for q in qs if getattr(q, "response", None) is not None)
            success_rate = ((completed-failed) / total * 100) if total > 0 else 0

            # Cerca la metrica principale per dare una rapida preview dei risultati
            avg_metrics_html = []
            main_metric = next((m for m in all_eval_fields if m in eval_fields), None)
            if total > 0 and main_metric:
                m_vals = [get_eval_value(q, main_metric) for q in qs]
                m_mean = mean([v for v in m_vals if isinstance(v, (int, float))])
                if m_mean is not None:
                    avg_metrics_html = [
                        html.Hr(style={"margin": "12px 0", "borderColor": "#e9ecef"}),
                        html.P([html.Strong(f"Avg {main_metric}:"), f" {m_mean:.4f}"], style={"margin": "0", "fontSize": "0.95em", "color": "#495057"})
                    ]

            # Crea la grafica della carta riepilogativa
            card = html.Div(
                style={
                    "border": "1px solid #dee2e6", "borderRadius": "8px", "padding": "18px",
                    "backgroundColor": "white", "minWidth": "240px", "flex": "1 1 240px",
                    "boxShadow": "0 4px 6px rgba(0,0,0,0.05)"
                },
                children=[
                             html.H4(ds, style={"marginTop": "0", "color": "#212529", "borderBottom": "2px solid #007bff", "paddingBottom": "8px", "marginBottom": "12px"}),
                             html.Div([
                                 html.Span("Totale Query:", style={"fontWeight": "600", "color": "#6c757d"}),
                                 html.Span(f" {total}", style={"float": "right", "fontWeight": "bold", "color": "#343a40"})
                             ], style={"marginBottom": "6px"}),
                             html.Div([
                                 html.Span("Completate (escluse partitioned?):", style={"fontWeight": "600", "color": "#6c757d"}),
                                 html.Span(f" {completed}", style={"float": "right", "fontWeight": "bold", "color": "#28a745"})
                             ], style={"marginBottom": "6px"}),
                             html.Div([
                                 html.Span("Fallite:", style={"fontWeight": "600", "color": "#6c757d"}),
                                 html.Span(f" {failed}", style={"float": "right", "fontWeight": "bold", "color": "#dc3545"})
                             ], style={"marginBottom": "6px"}),
                             html.Div([
                                 html.Span("Success Rate:", style={"fontWeight": "600", "color": "#6c757d"}),
                                 html.Span(f" {success_rate:.1f}%", style={"float": "right", "fontWeight": "bold", "color": "#17a2b8"})
                             ], style={"marginBottom": "8px"}),
                         ] + avg_metrics_html
            )
            cards.append(card)

        if not cards:
            return html.Div("Nessun dato disponibile per i filtri correnti.", style={"color": "#777", "fontStyle": "italic", "padding": "20px"})

        return html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "18px"},
            children=cards
        )


        # --- CALLBACK EXTRA: Calcolo e rendering del Fronte di Pareto ---
    @app.callback(
        Output("pareto-container", "children"),
        Input("set-tabs", "value"),
        Input("chart-type-tabs", "value"),
        Input("datasets-select", "value"),
        Input({"type": "filter", "field": ALL}, "value"),
        Input("suite-tabs", "value"),
        State({"type": "filter", "field": ALL}, "id"),
    )
    def update_pareto_front(active_set, chart_type, selected_datasets, filter_values, active_suite, filter_ids):
        if chart_type != "pareto":
            return dash.no_update

        if not active_set or active_set not in processed_suites.get(active_suite, {}):
            return html.Div("Nessun dato disponibile.")

        datasets = processed_suites[active_suite][active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_suite][active_set]
        selected_datasets = selected_datasets or dataset_names

        # Scegliamo la metrica (priorità ndcg_k, altrimenti la prima disponibile)
        metric_name = "ndcg_k" if "ndcg_k" in eval_fields else (eval_fields[0] if eval_fields else None)
        if not metric_name:
            return html.Div("Nessuna metrica di valutazione trovata per l'asse Y.")

        # Ricostruzione dei filtri correnti
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

        # Raggruppamento dei dati: (run_type, model, expr, prompt_level)
        groups = {}
        for ds in selected_datasets: # ds = model
            qs = [q for q in datasets[ds] if matches_filters(q) and getattr(q, "response", None) is not None]
            for q in qs:
                run_type = get_param_value(q, "run_type")
                expr = get_param_value(q, "expr")
                prompt_level = get_param_value(q, "prompt_level")

                combo_key = (str(run_type), ds, str(expr), str(prompt_level))

                # Preleviamo token e ndcg
                tokens = getattr(q, "tokens", None)
                val = get_eval_value(q, metric_name)

                if isinstance(tokens, (int, float)) and isinstance(val, (int, float)):
                    if combo_key not in groups:
                        groups[combo_key] = {'tokens': [], 'ndcg': [], 'model': ds}
                    groups[combo_key]['tokens'].append(float(tokens))
                    groups[combo_key]['ndcg'].append(float(val))

        if not groups:
            return html.Div("Dati insufficienti o mancanti (verifica che 'q.tokens' e la metrica esistano).")

        # Calcolo delle medie per ogni combinazione
        points = []
        for combo, data in groups.items():
            t_mean = sum(data['tokens']) / len(data['tokens'])
            n_mean = sum(data['ndcg']) / len(data['ndcg'])
            hover_txt = f"Model: {combo[1]}<br>Run Type: {combo[0]}<br>Expr: {combo[2]}<br>Prompt: {combo[3]}<br>Tokens: {t_mean:.1f}<br>{metric_name}: {n_mean:.4f}"
            points.append({
                'tokens': t_mean, 'ndcg': n_mean, 'model': data['model'], 'hover': hover_txt
            })

        # Calcolo del Fronte di Pareto (ordinamento per tokens crescente, teniamo i massimi di ndcg)
        points.sort(key=lambda x: x['tokens'])
        pareto_front = []
        max_ndcg = -float('inf')

        for p in points:
            if p['ndcg'] > max_ndcg:
                pareto_front.append(p)
                max_ndcg = p['ndcg']

        # Disegno del grafico
        fig = go.Figure()

        # Tracciamo una linea che unisce i punti della frontiera
        if pareto_front:
            fig.add_trace(go.Scatter(
                x=[p['tokens'] for p in pareto_front],
                y=[p['ndcg'] for p in pareto_front],
                mode="lines",
                line=dict(color='gray', width=2, dash='dash'),
                name="Pareto Frontier",
                hoverinfo="skip"
            ))

        # Tracciamo i punti divisi per modello iterando su selected_datasets
        # Questo garantisce che l'ordine della legenda e l'assegnazione dei colori
        # siano IDENTICI a quelli degli altri grafici!
        for model in selected_datasets:
            model_points = [p for p in points if p['model'] == model]

            # Se un modello è stato filtrato e non ha punti, non lo aggiungiamo
            if not model_points:
                continue

            fig.add_trace(go.Scatter(
                x=[p['tokens'] for p in model_points],
                y=[p['ndcg'] for p in model_points],
                mode="markers",
                marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
                name=model,
                text=[p['hover'] for p in model_points],
                hoverinfo="text"
            ))

        # Applicazione degli stili (layout e font aggiornati)
        fig.update_layout(
            # Aggiunto il totale dei punti nel titolo calcolando la lunghezza della lista 'points'
            title={"text": f"<b>Pareto Front (Tokens vs {metric_name}) - Totale Punti: {len(points)}</b>", "x": 0.5, "y": 0.97, "xanchor": "center", "font": {"size": font}},
            margin={"l": 80, "r": 20, "t": 60, "b": 160},
            height=750,
            legend={
                "orientation": "h",
                "xanchor": "center",
                "x": 0.5,
                "yanchor": "top",
                "y": -0.15,
                "font": {"size": 20}
            },
        )

        fig.update_xaxes(title_text="Avg Tokens", rangemode="tozero", tickfont={"size": tickfont}, title_font={"size": font})
        fig.update_yaxes(title_text=f"Avg {metric_name}", rangemode="tozero", tickfont={"size": tickfont}, title_font={"size": font})

        return dcc.Graph(
            figure=fig,
            config=get_global_config("pareto_front"),
            # maxWidth aumentato a 1300px per dare lo spazio fisico alla legenda di distendersi
            style={"height": "750px", "width": "100%", "maxWidth": "1300px", "margin": "0 auto"}
        )


    # --- CALLBACK 4: Aggiorna i grafici Plotly ---
    @app.callback(
        Output({"type": "metric-graph", "metric": all_eval_fields[0]}, "figure"),
        *[Output({"type": "metric-graph", "metric": m}, "figure") for m in all_eval_fields[1:]],
        Input("set-tabs", "value"),
        Input("chart-type-tabs", "value"),
        Input("datasets-select", "value"),
        Input("x-fields-select", "value"),
        Input({"type": "filter", "field": ALL}, "value"),
        Input("suite-tabs", "value"),
        State({"type": "filter", "field": ALL}, "id"),
    )
    def update_figures(active_set, chart_type, selected_datasets, x_fields, filter_values, active_suite, filter_ids):
        # Se siamo nel tab di riepilogo, non sprechiamo risorse calcolando i grafici
        if chart_type == "summary":
            return tuple([dash.no_update] * len(all_eval_fields))

        if not active_set or active_set not in processed_suites.get(active_suite, {}):
            return tuple([dash.no_update] * len(all_eval_fields))

        datasets = processed_suites[active_suite][active_set]
        dataset_names, param_fields, eval_fields, values_by_param = schemas[active_suite][active_set]

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
            _qs = [q for q in datasets[ds] if matches_filters(q)]
            def q_sort_key(q):
                return tuple(safe_sort_val(get_param_value(q, f)) for f in x_fields)
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

            # Traccia fantasma per allineare gli assi multicategorici in Plotly
            if all_keys:
                if chart_type == "line":
                    fig.add_trace(go.Scatter(x=x_plotly, y=[None]*len(all_keys), showlegend=False, hoverinfo="skip"))
                elif chart_type == "box":
                    fig.add_trace(go.Box(x=x_plotly, y=[None]*len(all_keys), showlegend=False, hoverinfo="skip"))

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

                    fig.add_trace(go.Scatter(
                        x=x_plotly, y=y_aligned,
                        mode="markers", # <--- Cambiato da "lines+markers" a "markers"
                        name=ds,
                        marker=dict(
                            color=GLOBAL_COLOR_MAP.get(ds, "#000000"),
                            size=10 # <--- Aggiunto size=10 per rendere i punti belli visibili ora che non c'è la linea
                        )
                    ))

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

                    x_plot_box = [ds_x_outer, ds_x_inner] if len(x_fields) == 2 else ds_x_single
                    fig.add_trace(go.Box(
                        x=x_plot_box, y=ds_y, name=ds, boxpoints="outliers",
                        marker=dict(color=GLOBAL_COLOR_MAP.get(ds, "#000000"))   # <--- COLORE FISSO
                    ))

                elif chart_type == "heatmap":
                    # --- ANCHE QUI IL TRUCCO SALVAVITA ---
                    # Eseguiamo il calcolo della matrice solo al primo giro.
                    # Evita di calcolare le correlazioni e sovrapporre la heatmap N volte!
                    if ds != selected_datasets[0]:
                        continue
                    z_data = []
                    y_labels = []

                    active_values = {p: set() for p in param_fields}
                    for tds in selected_datasets:
                        for q in filtered_by_ds[tds]:
                            for p in param_fields:
                                active_values[p].add(get_param_value(q, p))

                    expanded_features = []
                    for p in param_fields:
                        if len(active_values[p]) <= 1:
                            continue

                        sample_val = values_by_param[p][0] if values_by_param[p] else None
                        if isinstance(sample_val, str):
                            for v in values_by_param[p]:
                                if v in active_values[p]:
                                    expanded_features.append(f"{p} == {v}")
                        else:
                            expanded_features.append(p)

                    if not expanded_features:
                        expanded_features = ["Nessun parametro variabile"]

                    for tds in selected_datasets:
                        tqs = filtered_by_ds[tds]
                        if not tqs:
                            continue

                        df_dict = {f: [] for f in expanded_features}
                        m_vals = []

                        for q in tqs:
                            m_val = get_eval_value(q, metric_name)
                            m_vals.append(float(m_val) if isinstance(m_val, (int, float)) else np.nan)

                            if expanded_features == ["Nessun parametro variabile"]:
                                df_dict["Nessun parametro variabile"].append(0)
                                continue

                            for p in param_fields:
                                if len(active_values[p]) <= 1:
                                    continue

                                v = get_param_value(q, p)
                                sample_val = values_by_param[p][0] if values_by_param[p] else None

                                if isinstance(sample_val, str):
                                    for possible_v in values_by_param[p]:
                                        if possible_v in active_values[p]:
                                            df_dict[f"{p} == {possible_v}"].append(1 if v == possible_v else 0)
                                else:
                                    df_dict[p].append(v)

                        df_dict[metric_name] = m_vals
                        df = pd.DataFrame(df_dict)

                        corrs = []
                        for f in expanded_features:
                            if f == "Nessun parametro variabile":
                                corrs.append(0.0)
                            elif df[f].nunique() > 1 and df[metric_name].nunique() > 1:
                                c = df[f].corr(df[metric_name], method='spearman')
                                corrs.append(c if not pd.isna(c) else 0.0)
                            else:
                                corrs.append(0.0)

                        z_data.append(corrs)
                        y_labels.append(tds)

                    if z_data:
                        fig.add_trace(go.Heatmap(
                            z=z_data, x=expanded_features, y=y_labels,
                            colorscale='RdBu', zmin=-1, zmax=1, zmid=0,
                            text=[[f"{val:.2f}" for val in row] for row in z_data],
                            texttemplate="%{text}", hoverinfo="x+y+z",
                            # Colorbar spostata più vicina alla matrice
                            colorbar={"tickfont": {"size": tickfont}, "x": 1.02}
                        ))
                elif chart_type == "bar_corr":
                    # --- IL TRUCCO SALVAVITA ---
                    # Eseguiamo questo blocco solo per il primo modello del ciclo esterno.
                    # Altrimenti disegnerebbe il grafico N volte, creando barre multiple!
                    if ds != selected_datasets[0]:
                        continue
                    target_field = x_fields[0]

                    active_vals = set()
                    for tds in selected_datasets:
                        for q in filtered_by_ds[tds]:
                            active_vals.add(get_param_value(q, target_field))

                    active_vals = sorted(list(active_vals), key=safe_sort_val)

                    if len(active_vals) == 2:
                        # --- INVERSIONE VALORI ---
                        # Di default l'ordine (alfabetico) è formula, instruct.
                        # Assegnando active_vals[1] a val1, forziamo l'inversione
                        # così il calcolo sarà "formula over instruct"
                        val1, val2 = active_vals[1], active_vals[0]

                        bar_data = []

                        for tds in selected_datasets:
                            tqs = filtered_by_ds[tds]
                            if not tqs: continue

                            x_vals = []
                            y_vals = []
                            for q in tqs:
                                pv = get_param_value(q, target_field)
                                mv = get_eval_value(q, metric_name)
                                if isinstance(mv, (int, float)) and pv in (val1, val2):
                                    x_vals.append(1 if pv == val2 else 0)
                                    y_vals.append(float(mv))

                            if len(set(x_vals)) > 1 and len(set(y_vals)) > 1:
                                df_temp = pd.DataFrame({'x': x_vals, 'y': y_vals})
                                corr = df_temp['x'].corr(df_temp['y'], method='spearman')
                                if not pd.isna(corr):
                                    bar_data.append({'model': tds, 'corr': corr})

                        if bar_data:
                            bar_data.sort(key=lambda item: item['corr'], reverse=True)

                            models = [d['model'] for d in bar_data]
                            corrs = [d['corr'] for d in bar_data]

                            # RIGA ELIMINATA: colors = [GLOBAL_COLOR_MAP.get(m, "#1f77b4") for m in models]

                            fig.add_trace(go.Bar(
                                x=corrs, y=models,
                                orientation='h',
                                # --- COLORE UNICO SOBRIO ---
                                # Puoi usare un grigio scuro accademico come "#5D6D7E", oppure "gray", o "black"
                                marker=dict(color="#5D6D7E"),
                                text=[f"{c:.2f}" for c in corrs],
                                textposition='auto',
                                textfont=dict(size=16),
                                showlegend=False
                            ))
                            fig.add_vline(x=0, line_width=1.5, line_color="black")

            # --- SETUP LAYOUT FINALE ---
            layout_kwargs = {
                "title": {"text": f"<b>{metric_name}</b>", "x": 0.5, "y": 0.97, "xanchor": "center", "font": {"size": font}},
                "margin": {"l": 80, "r": 20, "t": 20, "b": 130},
                "height": 450,
                "legend": {
                    "orientation": "h", "xanchor": "center", "x": 0.5,
                    "yanchor": "top", "y": -0.22, "font": {"size": font}
                },
            }

            if chart_type == "box":
                layout_kwargs["boxmode"] = "group"
            elif chart_type == "heatmap":
                layout_kwargs["margin"] = {"l": 160, "r": 160, "t": 20, "b": 130}
                # (Mantieni qui il codice dell'annotazione "effect on ndcg" che avevi prima)
            elif chart_type == "bar_corr":
                # Margine sinistro molto grande (250) per contenere i nomi dei modelli
                layout_kwargs["margin"] = {"l": 280, "r": 50, "t": 60, "b": 100}
                layout_kwargs["showlegend"] = False # Nascondiamo la legenda orizzontale

            fig.update_layout(**layout_kwargs)

            # --- INGRANDIMENTO FONT ASSI E INSERIMENTO TITOLO Y ---
            if chart_type == "heatmap":
                fig.update_xaxes(type="category", tickfont={"size": tickfont}, title_font={"size": font})
                fig.update_yaxes(title_text="", autorange="reversed", tickfont={"size": tickfont}, side="left")
                fig.update_traces(textfont={"size": tickfont})
            elif chart_type == "bar_corr":
                # Aggiunto autorange="reversed" e categoryorder="trace" per rispettare l'ordine del filtro
                fig.update_yaxes(tickfont={"size": tickfont}, autorange="reversed", categoryorder="trace")

                if 'val1' in locals() and 'val2' in locals():
                    #fig.update_layout(title={"text": f"<b>{target_field}: {val2} vs {val1}</b>", "x": 0.5, "y": 0.95, "xanchor": "center", "font": {"size": font}})
                    fig.update_xaxes(title_text=f"{metric_name} variation: {val2} vs {val1}", tickfont={"size": tickfont}, title_font={"size": font})
                else:
                    fig.update_layout(title={"text": "<b>Seleziona un parametro (X) con esattamente 2 valori</b>", "x": 0.5, "y": 0.5, "xanchor": "center"})
            else:
                if len(x_fields) == 2:
                    fig.update_xaxes(type="multicategory", categoryorder="trace", tickfont={"size": tickfont}, title_font={"size": font})
                else:
                    fig.update_xaxes(type="category", categoryorder="trace", tickfont={"size": tickfont}, title_font={"size": font})

                # Senza grassetto come concordato!
                fig.update_yaxes(title_text=metric_name, rangemode="tozero", tickfont={"size": tickfont}, title_font={"size": font}, side="left")

            figures.append(fig)

        return tuple(figures)

    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    import experimenter

    run_dashboard(experiment_suites=experimenter.prepare_for_dashboard())