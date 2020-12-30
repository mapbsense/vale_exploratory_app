import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def layout(data):
    return dbc.Table.from_dataframe(data, striped=True, bordered=True, hover=True, responsive=True)  # id='df_data'
