import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


def scatter3d_ex(df, selected_features):
    df = df.sample(1000)
    fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1], z=selected_features[2],
                        color='labels')
    return fig


def bar_ex(df, lbl='labels', col='Z', y_='v'):
    bar = df.groupby([lbl, col], as_index=False).mean()
    return px.bar(bar, x=lbl, y=y_, color=col)


def dendrogram(df):
    den = df.groupby(['labels'], as_index=False)[['u', 'v', 'w']].mean()
    X = np.array(den[['labels', 'u', 'v', 'w']])
    fig = ff.create_dendrogram(X)
    fig.update_layout(
        autosize=True,
        margin=dict(
            l=5,
            r=5,
            b=10,
            t=10,
            pad=5
        )
    )
    return fig
