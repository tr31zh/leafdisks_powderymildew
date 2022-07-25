from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio


def plot_rejected_hist(df_src):
    fig = px.histogram(
        data_frame=df_src.sort_values(["comment"]),
        x="comment",
        color="comment",
        width=1200,
        height=400,
        text_auto=True,
    ).update_layout(
        font=dict(
            family="Courier New, monospace",
            size=14,
        ),
        xaxis=go.XAxis(title="Time", showticklabels=False),
        xaxis_visible=False,
    )

    fig.add_annotation(
        dict(
            # font=dict(color="yellow", size=15),
            x=0,
            y=-0.12,
            showarrow=False,
            text=f"From {df_src.shape[0]} sheets available {df_src[df_src.comment == 'success'].shape[0]} were successfully loaded",
            textangle=0,
            xanchor="left",
            xref="paper",
            yref="paper",
        )
    )

    return fig


def plot_oiv_homogeneity(df_src, oiv, width, height):
    return px.imshow(
        df_src[df_src.oiv == oiv].drop_duplicates().reset_index(drop=True),
        color_continuous_scale=px.colors.sequential.BuPu,
        height=height,
        width=width,
        title=f"OIV {oiv}",
    )


def plot_avg_by_oiv(df_src, width, height):
    return px.imshow(
        df_src.groupby(["oiv"]).mean().reset_index(drop=False),
        color_continuous_scale=px.colors.sequential.Viridis,
        height=height,
        width=width,
        title="Average values for all OIVs",
        text_auto=True,
    )


def plot_model(X, x_comp, y_comp, color, width, height, title):
    fig = px.scatter(
        x=X[:, x_comp],
        y=X[:, y_comp],
        color=color,
        height=height,
        width=width,
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )

    fig.update_xaxes(
        range=[
            X[:, x_comp].min() - 0.5,
            X[:, x_comp].max() + 0.5,
        ]
    )
    fig.update_yaxes(
        range=[
            X[:, y_comp].min() - 0.5,
            X[:, y_comp].max() + 0.5,
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"PCA {x_comp + 1}",
        yaxis_title=f"PCA {y_comp + 2}",
        legend_title="OIV value",
    )
    return fig
