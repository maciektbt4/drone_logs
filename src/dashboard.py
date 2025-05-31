# src/dashboard.py

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from pathlib import Path

# ─── 0) Ścieżki bazowe ───────────────────────────────────────────────
OUTPUT_DIR = Path("../output")   # katalog, w którym parse_log.py wygenerował foldery z runami

# ─── 1) Lista dostępnych runów ───────────────────────────────────────
def list_runs():
    if not OUTPUT_DIR.exists():
        return []
    return sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()])

# ─── 2) Layout dla pojedynczego runu ─────────────────────────────────
def layout_for_run(run_name: str):
    run_folder = OUTPUT_DIR / run_name
    if not run_folder.exists() or not run_folder.is_dir():
        return html.Div([
            html.H3(f"❌ Run '{run_name}' nie znaleziony."),
            html.P("Upewnij się, że najpierw uruchomiłeś parse_log.py."),
            dcc.Link("← Powrót do listy runów", href="/")
        ], style={"margin": "20px", "font-family": "Arial, sans-serif"})

    # ─── 2.1) Tabela z konfiguracją (jeśli istnieje config.csv) ───────
    cfg_path = run_folder / "config.csv"
    config_table = None
    if cfg_path.exists():
        df_cfg = pd.read_csv(cfg_path)
        config_table = dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_cfg.columns],
            data=df_cfg.to_dict("records"),
            page_size=len(df_cfg),
            style_table={"width": "50%", "overflowX": "auto", "margin": "10px 0"},
            style_cell={"textAlign": "left"}
        )

    # ─── 2.2) Wczytanie logów: trainlog.csv i best_results.csv ────────
    df_all  = pd.read_csv(run_folder / "trainlog.csv")
    df_best = pd.read_csv(run_folder / "best_results.csv")

    # ─── 2.3) Metryki globalne ────────────────────────────────────────
    total_time_hours   = df_all["t"].sum() / 3600
    total_best_success = int((df_best["Reward"] >= 100).sum())

    # ─── 2.4) Dane blokowe (co 10 000 kroków) ────────────────────────
    block_size = 10_000
    df_all["Step_block"] = (df_all["Step"] // block_size) * block_size

    avg_t_per_block = (
        df_all.groupby("Step_block")["t"]
              .mean()
              .reset_index(name="avg_t")
    )

    successes_per_block = (
        df_all.assign(success=(df_all["Reward"] >= 100))
              .groupby("Step_block")["success"]
              .sum()
              .reset_index(name="successes")
    )

    episodes_per_block = (
        df_all.groupby("Step_block")["Episode"]
              .nunique()
              .reset_index(name="episodes")
    )

    metrics_block = pd.merge(successes_per_block, episodes_per_block, on="Step_block") \
                     .melt(
                         id_vars="Step_block",
                         value_vars=["successes", "episodes"],
                         var_name="metric",
                         value_name="count"
                     )

    # ─── 2.5) Tworzymy listę elementów do umieszczenia na stronie runu ─
    components = []

    # Nagłówek + link powrotny
    components.append(
        html.Div([
            html.H2(f"🔍 Dashboard run: {run_name}"),
            dcc.Link("← Powrót do listy runów", href="/")
        ], style={"margin": "20px", "font-family": "Arial, sans-serif"})
    )

    # Jeśli istnieje config_table, umieszczamy ją w <details>
    if config_table is not None:
        components.append(
            html.Details(
                [
                    html.Summary("⚙️ Parametry z plików .cfg", 
                                 style={"font-weight": "bold", "cursor": "pointer"}),
                    config_table
                ],
                style={"width": "50%", "margin": "10px 0", "font-family": "Arial, sans-serif"}
            )
        )

    # Metryki globalne
    components.append(
        html.Div([
            html.P(f"🕒 Łączny czas treningu: {total_time_hours:.2f} h"),
            html.P(f"✅ Łączna liczba sukcesów (Reward ≥ 100): {total_best_success}")
        ], style={"margin": "20px", "font-family": "Arial, sans-serif"})
    )

    # Wykres: Czas kroku (t) vs Step
    fig_time = px.line(
        df_all, x="Step", y="t",
        title="Czas kroku (t) vs numer kroku",
        labels={"t": "Czas [s]"}
    )
    components.append(dcc.Graph(figure=fig_time))

    # Wykres: Ret vs Step
    fig_ret = px.line(
        df_all, x="Step", y="Ret",
        title="Ret w czasie",
        labels={"Ret": "Return"}
    )
    components.append(dcc.Graph(figure=fig_ret))

    # Tabela: Top 100 rekordów (najwyższy Ret)
    components.append(html.H4("🏆 Top 100 rekordów (największy Ret)"))
    components.append(
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df_best.columns],
            data=df_best.nlargest(100, "Ret").to_dict("records"),
            page_size=20,
            style_table={"overflowX": "auto", "margin": "10px 0"}
        )
    )

    # Wykres: Średni czas kroku per blok 10k
    fig_avg_t_block = px.line(
        avg_t_per_block,
        x="Step_block", y="avg_t",
        title=f"Średni czas kroku vs blok co {block_size} kroków",
        labels={"Step_block": "Krok (blok)", "avg_t": "Średni czas [s]"},
        markers=True
    )
    components.append(dcc.Graph(figure=fig_avg_t_block))

    # Wykres słupkowy: Sukcesy vs Epizody per blok 10k
    fig_bar = px.bar(
        metrics_block,
        x="Step_block", y="count", color="metric",
        barmode="group",
        title="Sukcesy (Reward ≥100) vs Liczba Epizodów na blok 10 000 kroków",
        labels={"Step_block": "Krok (blok)", "count": "Liczba", "metric": "Metryka"}
    )
    components.append(html.H4(f"Sukcesy i Epizody co {block_size} kroków"))
    components.append(dcc.Graph(figure=fig_bar))

    return html.Div(children=components, style={"font-family": "Arial, sans-serif"})


# ─── 3) Aplikacja Dash z routingiem ───────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


# ─── 4) Callback wyświetlający stronę główną lub odpowiedni run ─────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    # ─ Strona główna "/": pokazujemy tabelkę z dwoma kolumnami
    if pathname == "/" or pathname == "":
        runs = list_runs()
        if not runs:
            return html.Div([
                html.H3("Brak żadnych runów w katalogu 'output/'."),
                html.P("Uruchom parse_log.py, aby wygenerować dane.")
            ], style={"margin": "20px", "font-family": "Arial, sans-serif"})

        # Rozdzielamy runy według substringu "MobileNet" i "EfficientNet"
        mobile_runs    = [r for r in runs if "MobileNet" in r]
        efficient_runs = [r for r in runs if "EfficientNet" in r]

        # Obliczamy maksymalną długość, by wyrównać liczbę wierszy
        max_len = max(len(mobile_runs), len(efficient_runs))

        # Tworzymy kolejne wiersze tabeli: (mobilenet_run, efficientnet_run) lub puste
        table_rows = []
        for i in range(max_len):
            m_run = mobile_runs[i] if i < len(mobile_runs) else ""
            e_run = efficient_runs[i] if i < len(efficient_runs) else ""
            # Komórki z linkami (jeśli nazwa nie jest pusta)
            cell_mobile = dcc.Link(m_run, href=f"/run/{m_run}") if m_run else ""
            cell_effic = dcc.Link(e_run, href=f"/run/{e_run}") if e_run else ""
            table_rows.append(
                html.Tr([
                    html.Td(cell_mobile, style={"padding":"8px", "border":"1px solid #ddd"}),
                    html.Td(cell_effic, style={"padding":"8px", "border":"1px solid #ddd"})
                ])
            )

        # Nagłówki kolumn
        header_row = html.Tr([
            html.Th("MobileNet runs", style={"padding":"10px", "border":"1px solid #ddd"}),
            html.Th("EfficientNet runs", style={"padding":"10px", "border":"1px solid #ddd"})
        ])

        table = html.Table(
            [header_row] + table_rows,
            style={
                "borderCollapse": "collapse",
                "width": "80%",
                "margin": "20px auto",
                "font-family": "Arial, sans-serif"
            }
        )

        return html.Div([
            html.H2("📂 Dostępne runy:"),
            table
        ], style={"text-align": "center"})

    # ─ Strona dla konkretnego runu "/run/<run_name>"
    if pathname.startswith("/run/"):
        run_name = pathname.split("/run/")[-1]
        return layout_for_run(run_name)

    # ─ Inne ścieżki: 404
    return html.Div([
        html.H3("❌ 404: Strona nie znaleziona"),
        dcc.Link("← Powrót do listy runów", href="/")
    ], style={"margin": "20px", "font-family": "Arial, sans-serif"})


if __name__ == "__main__":
    app.run(debug=True)
