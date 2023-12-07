import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

#load_figure_template('journal')

app = Dash(
    __name__, 
    use_pages=True,
    external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME]
    )

app.config.suppress_callback_exceptions=True

SIDEBAR_STYLE = {
    "position": "fixed",
    "width": "203px",
    "padding": "4px 32px",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "zIndex": 999
}

CONTENT_STYLE = {
    "margin-left": "195px",
    "margin-right": "5px",
    "padding": "4px 16px",
}

sidebar = html.Div(
    [
        html.H3("Antimicrobial Consumption Dashboard"),
        html.H4(
            "Westmead Hospital 2023"
        ),
        dbc.Nav(
        [
            dbc.NavItem(
                dbc.NavLink([
                     html.Div(page["name"])
                ]
                , href=page["path"],
                active = "exact"
                )
            )
            for page in dash.page_registry.values()
        ],
        pills= True,
        vertical = True,
    )
    ],
    style = SIDEBAR_STYLE, className = "bg-light",
)

app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    dcc.Store(id='memory', storage_type = 'local'), # memory to store antibiotic order data
    dcc.Store(id='memory2', storage_type = 'local'), # memory2 to store antibiotic resistance data
    sidebar,
    dash.page_container
], style = CONTENT_STYLE
)

if __name__ == '__main__':
    app.run(debug=True)