# src/dashboard.py

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from pathlib import Path

# ─── 0) Ścieżki bazowe ─────────────────────────────────────────────
OUTPUT_DIR = Path("../output")   # katalog, który powstał po uruchomieniu parse_log.py

# ─── 1) Znajdź wszystkie dostępne runy (nazwy podfolderów w OUTPUT_DIR) ─
def list_runs():
    """
    Zwróć posortowaną listę nazw podfolderów w OUTPUT_DIR.
    Jeśli OUTPUT_DIR nie istnieje, zwróć pustą listę.
    """
    if not OUTPUT_DIR.exists():
        return []
    return sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()])

# ─── 2) Funkcja zwracająca layout dla konkretnego runu ────────────────
def layout_for_run(run_name: str):
    """
    Dla zadanego run_name (np. "run1") wczytaj jego CSV-e i stwórz 
    listę komponentów Dash (wykresy i tabele).
    Jeśli run nie istnieje, zwróć komunikat o błędzie.
    """
    run_folder = OUTPUT_DIR / run_name
    if not run_folder.exists() or not run_folder.is_dir():
        return html.Div([
            html.H3(f"❌ Run '{run_name}' nie znaleziony."),
            html.P("Upewnij się, że najpierw uruchomiłeś parse_log.py "
                   "i że taki folder istnieje w 'output/'."),
            dcc.Link("← Powrót do listy runów", href="/")
        ], style={"margin":"20px", "font-family":"Arial, sans-serif"})

    # 2.1 Wczytaj CSV-e z tego runu
    df_all  = pd.read_csv(run_folder / "trainlog.csv")
    df_best = pd.read_csv(run_folder / "best_results.csv")

    # 2.2 Oblicz metryki globalne
    total_time_hours   = df_all["t"].sum() / 3600
    total_best_success = int((df_best["Reward"] >= 100).sum())

    # 2.3 Agregacje do wykresów blokowych
    block_size = 10_000
    df_all["Step_block"] = (df_all["Step"] // block_size) * block_size

    # 2.3.1 Średni czas kroku
    avg_t_per_block = (
        df_all.groupby("Step_block")["t"]
              .mean()
              .reset_index(name="avg_t")
    )
    # 2.3.2 Sukcesy (Reward>=100)
    successes_per_block = (
        df_all.assign(success = df_all["Reward"] >= 100)
              .groupby("Step_block")["success"]
              .sum()
              .reset_index(name="successes")
    )
    # 2.3.3 Epizody unikalne
    episodes_per_block = (
        df_all.groupby("Step_block")["Episode"]
              .nunique()
              .reset_index(name="episodes")
    )
    # 2.3.4 Połącz do formatu „melt”, by mając dwa słupki obok siebie
    metrics_block = pd.merge(successes_per_block, episodes_per_block, on="Step_block") \
                     .melt(id_vars="Step_block",
                           value_vars=["successes", "episodes"],
                           var_name="metric",
                           value_name="count")

    # 2.4 Budujemy layout (lista komponentów)
    components = []

    # Nagłówek + link powrotny
    components.append(html.Div([
        html.H2(f"🔍 Dashboard run: {run_name}"),
        dcc.Link("← Powrót do listy runów", href="/")
    ], style={"margin":"20px", "font-family":"Arial, sans-serif"}))

    # Metryki globalne
    components.append(html.Div([
        html.P(f"🕒 Łączny czas treningu: {total_time_hours:.2f} h"),
        html.P(f"✅ Łączna liczba sukcesów (Reward ≥ 100): {total_best_success}")
    ], style={"margin":"20px"}))

    # Wykres: Czas kroku vs Step
    fig_time = px.line(
        df_all, x="Step", y="t",
        title="Czas kroku (t) vs numer kroku",
        labels={"t":"Czas [s]"}
    )
    components.append(dcc.Graph(figure=fig_time))

    # Wykres: Ret vs Step
    fig_ret = px.line(
        df_all, x="Step", y="Ret",
        title="Ret w czasie",
        labels={"Ret":"Return"}
    )
    components.append(dcc.Graph(figure=fig_ret))

    # Tabela top 100 (najwyższy Ret)
    components.append(html.H4("🏆 Top 100 rekordów (największy Ret)"))
    components.append(dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df_best.columns],
        data=df_best.nlargest(100, "Ret").to_dict("records"),
        page_size=20,
        style_table={"overflowX": "auto", "margin":"20px"}
    ))

    # Wykres: Średni czas kroku na blok
    fig_avg_t_block = px.line(
        avg_t_per_block,
        x="Step_block", y="avg_t",
        title=f"Średni czas kroku vs blok co {block_size} kroków",
        labels={"Step_block":"Krok (blok)", "avg_t":"Średni czas [s]"},
        markers=True
    )
    components.append(dcc.Graph(figure=fig_avg_t_block))

    # Wykres słupkowy: Sukcesy vs Epizody na blok
    fig_bar = px.bar(
        metrics_block,
        x="Step_block", y="count", color="metric",
        barmode="group",
        title="Sukcesy (Reward ≥100) vs Liczba epizodów na blok 10 000 kroków",
        labels={"Step_block":"Krok (blok)", "count":"Liczba", "metric":"Metryka"}
    )
    components.append(html.H4(f"Sukcesy i Epizody co {block_size} kroków"))
    components.append(dcc.Graph(figure=fig_bar))

    return html.Div(children=components, style={"font-family":"Arial, sans-serif"})


# ─── 3) Budowa aplikacji Dash ─────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout root: nawigacja oparta na URL
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# ─── 4) Callback do routingu ──────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """
    Jeśli pathname = "/" → wrzuć layout index (lista runów).
    Jeśli pathname zaczyna się od "/run/", np. "/run/run1", 
       to weź run_name = "run1" i zwróć layout_for_run(run_name).
    W przeciwnym razie zwróć 404.
    """
    if pathname == "/" or pathname == "":
        # Strona główna: lista „runów” z linkami
        runs = list_runs()
        if not runs:
            return html.Div([
                html.H3("Brak żadnych runów w katalogu 'output/'."),
                html.P("Najpierw uruchom parse_log.py, aby wygenerować CSV.")
            ], style={"margin":"20px", "font-family":"Arial, sans-serif"})

        # Tworzymy listę linków do poszczególnych runów
        links = []
        for r in runs:
            links.append(html.Li(dcc.Link(r, href=f"/run/{r}")))

        return html.Div([
            html.H2("📂 Dostępne runy:"),
            html.Ul(links, style={"margin":"20px", "font-family":"Arial, sans-serif"})
        ])

    # Jeśli URL wygląda jak "/run/<run_name>"
    if pathname.startswith("/run/"):
        run_name = pathname.split("/run/")[-1]
        return layout_for_run(run_name)

    # Wszędzie indziej: 404
    return html.Div([
        html.H3("❌ 404: Strona nie znaleziona"),
        dcc.Link("← Powrót do listy runów", href="/")
    ], style={"margin":"20px", "font-family":"Arial, sans-serif"})


if __name__ == "__main__":
    app.run(debug=True)
