# #!/usr/bin/env python
import os
import urllib.parse as urlparse

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import render_template, send_from_directory
from jupyter_dash import JupyterDash

from utils_clust.fit_clusters import figure_data_w_model
from utils_io.parse_uploads import simple_merge, fn_to_df_, simple_concat, get_col_manes, gen_profile_from_df
from views.eda import scatter3d_ex, bar_ex, dendrogram

JupyterDash.infer_jupyter_proxy_config()
app = JupyterDash(__name__,
                  suppress_callback_exceptions=True,
                  external_stylesheets=[dbc.themes.MINTY])

FIT_DATASETS = os.listdir(r'datasets')
PRED_DATASETS = os.listdir(r'predictions')
RAW_DATASETS = os.listdir(r'raw_datasets')

dropdown_model = dbc.FormGroup([
    dbc.Label("With Model:", html_for="dropdown-av-models"),
    dcc.Dropdown(options=[{'label': model, 'value': model} for model in ['Kmeans', 'MiniSOM', 'OasisSOM']],
                 multi=False,
                 id="dropdown-av-models")
])
dropdown_data = dbc.FormGroup([
    dbc.Label("And Data:", html_for="dropdown-av-data"),
    dcc.Dropdown(
        id="dropdown-av-data",
        options=[{'label': ds.split('.')[0], 'value': ds.split('.')[0]} for ds in FIT_DATASETS],
    ),
])

n_clasesslider = dbc.FormGroup([
    dbc.Label("Nº Classes:", html_for="n-clss-slider"),
    dcc.Slider(min=2, max=12, marks={i: '{}'.format(i) for i in range(13)}, value=4, id='n-clss-slider')
])
form = dbc.Form([dropdown_model, dropdown_data, n_clasesslider,
                 dbc.Button("Fit", color="primary", outline=True, id='run-calculations'),
                 dbc.Button("More", color="info", outline=True, id='open-info', className='float-right')])

sidebar = html.Div([
    html.Img(src='https://upload.wikimedia.org/wikipedia/en/thumb/9/97/Vale_logo.svg/1920px-Vale_logo.svg.png',
             width="100%"),
    html.Hr(),
    html.H3("Inspect Table", className="lead"),
    dbc.Nav([
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem(ds, href=f"store?data={ds.split('.')[0]}") for ds in FIT_DATASETS],
            label="Dataset",
            nav=True,
            id='explore-datasets'
        ),
    ], vertical=True, pills=True),
    dbc.Nav([
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem(ds, href=f"pred?data={ds.split('.')[0]}") for ds in PRED_DATASETS],
            label="Predictions",
            nav=True,
            id='explore-predictions'
        ),
    ], vertical=True, pills=True),
    html.Hr(),
    html.H3("Cluster Analysis", className="lead"),
    form,
    html.Hr(),
    html.H3("Merge and Concatenate Files", className="lead"),
    dbc.NavLink('XYZ Data File', href='upload?folder_path=raw_dataset', target='_blank')],
    className='position-fixed p-3 m-0', style={"width": "20rem"})
content = dcc.Loading(html.Div(id="page-content", className='p-4', style={"margin-left": "20rem"}), id='loading-state')

ex_att = dbc.FormGroup([
    dbc.Label("Exclude Attributes:", html_for="exclude-cols", width=2),
    dbc.Col(
        dcc.Dropdown(
            id='exclude-cols',
            value=['X', 'Y', 'Z'],
            multi=True
        ),
        width=10,
    ),
], row=True)
cat_feat = dbc.FormGroup([
    dbc.Label("Set Categorical Attributes:", html_for="cat-cols", width=2),
    dbc.Col(
        dcc.Dropdown(
            id='cat-cols',
            # value=['X', 'Y', 'Z'],
            multi=True
        ),
        width=10,
    ),
], row=True)
prep_feat = dbc.FormGroup([
    dbc.Label("Preprocessing:", html_for="prep-feat", width=2),
    dbc.Col(
        dbc.Checklist(
            id="prep-feat",
            options=[
                {'label': 'Standardize Features', 'value': 'std'},
                {'label': 'Normal Transform (box-cox)', 'value': 'bocx'},
                {'label': 'Dimensionality Reduction (PCA) ', 'value': 'pca'}
            ]),
        width=10,
    ),
], row=True)
m_body = dbc.Form([
    ex_att,
    cat_feat,
    prep_feat
])
modal = dbc.Modal([
    dbc.ModalHeader("Adjust Clustering Parameters"),
    dbc.ModalBody(m_body),
    dbc.ModalFooter(dbc.Button("Done", id="close-info", className="ml-auto")),
], id="modal", size="lg")
index_cols = dbc.FormGroup([
    dbc.Label("Merge on Column Names", html_for="index-column", width=2),
    dbc.Col(
        dbc.Input(
            value='X,Y,Z',
            id="index-column",
            placeholder="Comma Separated Column Names to use as index on merge",
        ),
        width=10,
    ),
], row=True)
concat_to = dbc.FormGroup([
    dbc.Label("Concatenate to an existing dataset", html_for="concat-to", width=2),
    dbc.Col(
        dcc.Dropdown(
            id="concat-to",
            options=[{'label': ds.split('.')[0], 'value': ds} for ds in FIT_DATASETS],
        ),
        width=10,
    ),
], row=True)
merge_on_dropdown = dbc.FormGroup([
    dbc.Label("Join Type", html_for="join-radio-items"),
    dbc.RadioItems(
        value='inner',
        id="join-radio-items",
        options=[
            {"label": "Inner", "value": 'inner'},
            {"label": "Outer", "value": 'outer'},
        ],
    )])
drop_file = dbc.FormGroup([
    dbc.Label(html.H5("Data-file(s) to table *"), html_for="upload-data", width=2),
    dbc.Col(
        dcc.Dropdown(
            options=[{'label': ds.split('.')[0], 'value': ds} for ds in RAW_DATASETS],
            multi=True,
            id='upload-data',
        ),
        width=10,
    )
], row=True)
fill_na = dbc.FormGroup([
    dbc.Label("Fill Empty Spaces With Zeros", html_for="fill-na"),
    dbc.RadioItems(
        value=True,
        id="fill-na",
        options=[
            {"label": "Yes", "value": True},
            {"label": "No", "value": False}
        ],
    )])
name_df = dbc.FormGroup([
    dbc.Label(html.H5("Output table name *"), html_for="name-df", width=2),
    dbc.Col(
        dbc.Input(
            # value='X,Y,Z',
            id="name-df",
            # placeholder="Comma Separated Column Names to use as index on merge",
        ),
        width=10,
    ),
], row=True)
merge_btn = dbc.Button("Merge Data", id="merge-button", className="mt-3", color="info", outline=True, block=True)
upload_info = dbc.FormText(html.A('Read Documentation',
                                  href='https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html',
                                  target='_blank'))
collapse = dbc.Collapse([index_cols, dbc.Row([
    dbc.Col(
        merge_on_dropdown,
        width=6,
    ),
    dbc.Col(
        fill_na,
        width=6,
    ),
], form=True)], id='merge-collapse')
drop_file_form = dbc.Form([drop_file,
                           name_df,
                           collapse,
                           concat_to,
                           upload_info,
                           merge_btn,
                           html.Hr()])

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    modal

])
app.title = 'Vale Exploration Dashboard'


def build_preds_view(result, current_df):
    opts = [{'label': r, 'value': r} for r in result.columns]
    pred_opts = [{'label': r.split('.')[0] + ' predictions', 'value': r} for r in os.listdir(r'predictions')]
    avlb = dcc.Dropdown(options=pred_opts, value=current_df, clearable=False, id='pred-table')
    scatter_card = dbc.Card(dbc.CardBody([
        html.H5("Scatter Plot", className="card-title"),
        # html.P("This card has some text content, but not much else"),
        dcc.Graph(figure=scatter3d_ex(result, ['X', 'Y', 'Z'])),
        dbc.Row([
            dbc.Col([
                dbc.Label("X-Axis:", html_for="x-axis"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='x-axis', value='X', clearable=False)])],
                width=3),
            dbc.Col([
                dbc.Label("Y-Axis:", html_for="y-axis"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='y-axis', value='Y', clearable=False)])],
                width=3),
            dbc.Col([
                dbc.Label("Z-Axis:", html_for="z-axis"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='z-axis', value='Z', clearable=False)])],
                width=3),
            dbc.Col([
                dbc.Label("Color:", html_for="z-axis"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='scatter-color', value='Z', clearable=False)])],
                width=3)
        ]),
        dbc.Button("Update", color="primary", id='update-scatter', block=True, outline=True)
    ]))
    bar_card = dbc.Card(dbc.CardBody([
        html.H5("Bar plot", className="card-title"),
        # html.P("This plot is meant to visualize ..."),
        dcc.Graph(figure=bar_ex(result), id='bar-plot'),
        dbc.Row([
            dbc.Col([
                dbc.Label("Bars:", html_for="bar-feat"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='bar-feat', value='labels', clearable=False)])],
                width=4),
            dbc.Col([
                dbc.Label("Y-Axis:", html_for="y-ax-feat"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='y-ax-feat', value='v', clearable=False)])],
                width=4),
            dbc.Col([
                dbc.Label("Color:", html_for="color-feat"),
                dbc.FormGroup([dcc.Dropdown(options=opts, id='color-feat', value='Z', clearable=False)])],
                width=4)
        ]),
        dbc.Button("Update", color="primary", id='update-bar', block=True, outline=True)
    ]))
    dendrogram_card = dbc.Card(dbc.CardBody([
        html.H5("Dendrogram", className="card-title"),
        dcc.Graph(figure=dendrogram(result)),
    ]))
    return html.Div([dbc.Row(dbc.Col(avlb, width=12)),
                     dbc.Row([dbc.Col(scatter_card, width=6), dbc.Col(bar_card, width=6)], className='mt-3'),
                     dbc.Row(dbc.Col(dendrogram_card, width=12), className='mt-3')], id='build_preds_view')


@app.callback(Output("page-content", "children"),
              [Input("url", "href")])
def render_page_content(pathname):
    parsed = urlparse.urlparse(pathname)
    parsed_query = urlparse.parse_qs(parsed.query)
    if parsed.path == '//' or parsed.path is None:
        return "Welcome"
        # raise PreventUpdate

    elif parsed.path == "//store" or parsed.path == "//pred":
        from_dir = 'datasets' if parsed.path == "//store" else 'predictions'
        try:
            selected_df = parsed_query['data'][0]
            df, dfd = fn_to_df_(selected_df + '.csv', from_=from_dir, describe=True)
            tabl = dbc.Table.from_dataframe(dfd, striped=True, bordered=True, hover=True, responsive=True)

            if from_dir == 'predictions':
                selected_df += '_pred'

            if selected_df + '.html' not in os.listdir('templates'):
                gen_profile_from_df(df, selected_df)

            profile_link = dbc.CardLink('Check More Summary Statistics', external_link=True,
                                        href=f'temp/{selected_df}', target='_blank')
            download_pill = dbc.Badge("Download", href=f'{from_dir}/{selected_df}',
                                      color="primary", className="ml-3", external_link=True)
            df_title = html.H2([selected_df, download_pill])
            df_subtitle = html.H6("Descriptive Statistics", className="card-subtitle mt-1 mb-3")
            card = dbc.Card(dbc.CardBody([df_title, df_subtitle, tabl, profile_link]))
        except Exception as e:
            return dbc.Alert([html.H4(str(e), className="alert-heading")], color="info")
        return card

    elif parsed.path == "//predict":
        try:
            # TODO funcionality for several model (list of models)
            parsed_kwargs = {key: value[0] for (key, value) in parsed_query.items()}

            if 'data' not in parsed_kwargs or parsed_kwargs['data'] == 'None':
                raise RuntimeError('Please select the Data for training data')
            if 'model' not in parsed_kwargs or parsed_kwargs['model'] == 'None':
                raise RuntimeError('Please select the Model')

            result = figure_data_w_model(**parsed_kwargs)
            view = build_preds_view(result, parsed_kwargs['data'] + '.csv')

        except Exception as e:
            return dbc.Alert(str(e), color="warning"),
        return view

    elif parsed.path == '//upload':
        return drop_file_form, html.Div(id='output-data-upload')

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(Output("run-calculations", "href"),
              [Input("dropdown-av-models", "value"),
               Input("dropdown-av-data", "value"),
               Input("n-clss-slider", "value"),
               Input('exclude-cols', 'value'),
               Input('prep-feat', 'value'),
               Input('cat-cols', 'value')])
def run_calculations_url(av_models, av_data, n_clss, use_cols, prep_feat, cat_cols):
    url_ = f'predict?data={av_data}&model={av_models}&n_clusters={n_clss}'
    if use_cols:
        url_ += f'&drop_cols={",".join(use_cols)}'
    if cat_cols:
        url_ += '&encode'  # TODO: Improve action
    if prep_feat:
        if 'std' in prep_feat:
            url_ += '&std'
        # TODO: add if box-cox and PCA
    return url_


@app.callback(Output('output-data-upload', 'children'),
              [Input('merge-button', 'n_clicks')],
              [State('index-column', 'value'),
               State('name-df', 'value'),
               State('join-radio-items', 'value'),
               State('upload-data', 'value'),
               State('fill-na', 'value'),
               State('concat-to', 'value')])
def merge_dfs(run_merge, on_cols, df_name, how_t, list_of_names, fill_w_zeros, concat_to_):
    if run_merge is None:
        raise PreventUpdate
    else:
        try:
            if not list_of_names:
                raise RuntimeError('Please specify files to process ("Data-file(s) to table")')
            if not df_name:
                raise RuntimeError('Please specify an output-file name ("Output table name")')
            if df_name + '.csv' in os.listdir('datasets'):
                raise RuntimeError('The output-file name is already being used')
            if len(list_of_names) > 1:
                if not on_cols:
                    raise RuntimeError('Please specify column names for merging')
                on_cols = on_cols.split(',')
                dfs = [fn_to_df_(fn) for fn in list_of_names]
                ds = simple_merge(dfs, on_=on_cols, how_=how_t)
            else:
                ds = fn_to_df_(list_of_names[0])

            if concat_to_:
                df_c = fn_to_df_(concat_to_, from_='datasets')
                ds = simple_concat([df_c, ds])

            if fill_w_zeros:
                # ds = ds.dropna(how='all')
                # ds = ds.dropna(axis=1, how='all')
                ds = ds.fillna(0)

            ds.to_csv(f'datasets/{df_name}.csv', index=False)
        except Exception as e:
            return dbc.Alert([html.H4(str(e), className="alert-heading")], color="info")

    return dbc.Alert([html.H4("The data has been processed", className="alert-heading", id='effective-data-upload'),
                      html.P([f"You may now find it as '{df_name}'"])])


@app.callback([Output("open-info", "disabled"),
               Output("run-calculations", "disabled")],
              [Input("dropdown-av-data", "value"),
               Input("dropdown-av-models", "value")])
def enable_modal(data, model):
    act = False if data and model else True
    return act, act


@app.callback([Output("modal", "is_open"),
               Output('exclude-cols', 'options')],
              [Input("open-info", "n_clicks"),
               Input("close-info", "n_clicks")],
              [State("modal", "is_open"),
               State("dropdown-av-data", "value")])
def exclude_cols_options(click_open, click_close, state, data):
    if not data:
        raise PreventUpdate
    values = get_col_manes(data)
    options = [{'label': opt, 'value': opt} for opt in values]
    return not state, options


@app.callback([Output('concat-to', 'options'),
               Output('dropdown-av-data', 'options'),
               Output('explore-datasets', 'children')],
              [Input('effective-data-upload', 'children')])
def update_av_dataset(_):
    av_data = os.listdir('datasets')
    av_data_wext = [{'label': ds.split('.')[0], 'value': ds} for ds in av_data]
    av_data_noext = [{'label': ds.split('.')[0], 'value': ds.split('.')[0]} for ds in av_data]
    ds_to_exp = [dbc.DropdownMenuItem(fn, href=f"store?data={fn.split('.')[0]}") for fn in av_data]
    return av_data_wext, av_data_noext, ds_to_exp


@app.callback(Output('explore-predictions', 'children'),
              Input('build_preds_view', 'children'))
def update_av_predictions(_):
    av_pred = os.listdir('predictions')
    ds_pred = [dbc.DropdownMenuItem(fn, href=f"pred?data={fn.split('.')[0]}") for fn in av_pred]
    return ds_pred


@app.callback(Output('cat-cols', 'options'),
              [Input("open-info", "n_clicks"),
               Input('exclude-cols', 'value')],
              [State('dropdown-av-data', 'value')])
def cat_cols_options(open_info, excl_cls, data):
    if open_info:
        values = get_col_manes(data)
        options = [{'label': opt, 'value': opt} for opt in values if opt not in excl_cls]
        return options
    else:
        raise PreventUpdate


@app.callback(Output("merge-collapse", "is_open"),
              [Input('upload-data', 'value')])
def toggle_collapse(list_of_names):
    if list_of_names is None:
        raise PreventUpdate
    return len(list_of_names) > 1


@app.callback(Output("bar-plot", "figure"),
              [Input("update-bar", "n_clicks")],
              [State('pred-table', "value"),
               State('bar-feat', "value"),
               State('y-ax-feat', "value"),
               State('color-feat', "value")])
def exclude_cols_options(update, labeled_dfm, bar_feat, y_ax_feat, color_feat):
    if update is None:
        raise PreventUpdate
    labeled_dfm = fn_to_df_(labeled_dfm, from_='predictions')
    return bar_ex(labeled_dfm, lbl=bar_feat, col=color_feat, y_=y_ax_feat)


@app.server.route('//temp/<pagename>')
def serve_template(pagename):
    return render_template(pagename + '.html')


@app.server.route("/<path:dir_>/<path:fn_>")
def download(dir_, fn_):
    if dir_ in ['datasets', 'predictions']:
        if dir_ == 'predictions':
            fn_ = fn_.split('_')[0]
        return send_from_directory(dir_, fn_ + '.csv', as_attachment=True)


if __name__ == '__main__':
    app.run_server(debug=False)
