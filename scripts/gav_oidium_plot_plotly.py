import os

import numpy as np
import pandas as pd

import skimage.io as skio

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import gav_oidium_const as goc


def plot_sample_oiv_images(width=None, height=None):
    # return sorted(os.listdir(os.path.join(goc.datain_path, "images", "oiv_samples")))
    files = [
        os.path.join(goc.datain_path, "images", "oiv_samples", f)
        for f in [
            "smp_oiv_1.png",
            "smp_oiv_3.png",
            "smp_oiv_5.png",
            "smp_oiv_57.png",
            "smp_oiv_7.png",
            "smp_oiv_7_tn_diff.png",
            "smp_oiv_9.png",
            "smp_oiv_9_adp.png",
        ]
    ]
    # return [os.path.isfile(f) for f in files]
    titles = [
        "OIV 1",
        "OIV 3",
        "OIV 5",
        "OIV 5 or 7",
        "OIV 7",
        "OIV 7, different necrose size",
        "OIV 9",
        "OIV 9 bis",
    ]

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=np.array(titles).flatten(),
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    for idx, f in enumerate(files):
        fig.add_trace(
            go.Image(z=skio.imread(f)),
            row=1 if idx < 4 else 2,
            col=(idx % 4) + 1,
        )

    fig.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False
    ).update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def plot_inconsistencies(
    df,
    sort_values: bool = True,
    width=1400,
    height=1000,
    title=None,
    move_legend=True,
):
    columns = [
        ["sporulation", "densite_sporulation", ""],
        ["necrose", "surface_necrosee", "taille_necrose"],
        ["ligne", "colonne", "oiv"],
    ]

    fig = make_subplots(rows=3, cols=3, subplot_titles=np.array(columns).flatten())

    for idl, l in enumerate(columns):
        for idc, c in enumerate(l):
            if not c:
                continue
            fig.add_trace(
                go.Histogram(
                    x=df[c].sort_values().astype(str)
                    if sort_values is True
                    else df[c].astype(str),
                    texttemplate="%{y}",
                    textfont_size=20,
                    name=c,
                ),
                row=idl + 1,
                col=idc + 1,
            )

    fig.update_layout(
        height=height,
        width=width,
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    if move_legend is True:
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
            ),
        )

    return fig


def plot_variance(
    df_ev,
    height=700,
    width=800,
):
    df_ev = df_ev.assign(cumulative=df_ev["exp_var_per"].cumsum())
    ev_fig = go.Figure()
    ev_fig.add_trace(
        go.Bar(
            x=df_ev["pc"],
            y=df_ev["exp_var_per"],
            name="individual",
            texttemplate="%{y}",
            textfont_size=20,
        )
    )
    ev_fig.add_trace(
        go.Scatter(
            x=df_ev["pc"],
            y=df_ev["cumulative"],
            name="cumulative",
        )
    )
    ev_fig.update_layout(
        height=height,
        width=width,
        title="Explained variance by different principal components",
        xaxis_title="Principal component",
        yaxis_title="Explained variance in percent",
    )
    return ev_fig


def plot_rejected_hist(df_src):
    fig = px.histogram(
        data_frame=df_src.sort_values(["comment"]),
        x="comment",
        color="comment",
        width=1000,
        height=600,
        text_auto=True,
    ).update_layout(
        font=dict(
            family="Courier New, monospace",
            size=14,
        ),
        xaxis=go.XAxis(title="Time", showticklabels=False),
        xaxis_visible=False,
        legend=dict(xanchor="center", yanchor="top", y=-0.1, x=0.3),
    )

    fig.add_annotation(
        dict(
            # font=dict(color="yellow", size=15),
            x=0.1,
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


def plot_oiv_homogeneity(df_src, oiv, width=None, height=None):
    return px.imshow(
        df_src[df_src.oiv == oiv].drop_duplicates().reset_index(drop=True),
        color_continuous_scale=px.colors.sequential.BuPu,
        height=height,
        width=width,
        title=f"OIV {oiv}",
    )


def plot_all_homogeneity(df_src, width=None, height=None):
    oivs = [1, 3, 5, 7, 9]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"OIV {i}" for i in oivs] + ["Average values for all OIV"],
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    for i in oivs:
        fig.add_trace(
            go.Image(
                z=df_src[df_src.oiv == i].drop_duplicates().reset_index(drop=True)
            ),
            row=1 if i in [1, 3, 5] else 2,
            col=(i % 3) + 1,
        )

    fig.add_trace(
        go.Image(z=df_src.groupby(["oiv"]).mean().reset_index(drop=False)),
        row=2,
        col=3,
    )

    fig.update_xaxes(showticklabels=False).update_yaxes(
        showticklabels=False
    ).update_layout(
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def plot_avg_by_oiv(df_src, width=None, height=None):
    return px.imshow(
        df_src.groupby(["oiv"]).mean().reset_index(drop=False),
        color_continuous_scale=px.colors.sequential.Viridis,
        height=height,
        width=width,
        title="Average values for all OIVs",
        text_auto=True,
    )


def plot_pca(
    pca,
    df,
    pca_columns,
    color,
    title=" ",
    pcx=0,
    pcy=1,
    width=None,
    height=None,
    color_discrete_sequence=px.colors.qualitative.Dark2,
):
    components = pca.fit_transform(df)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 2

    fig = (
        px.scatter(
            x=components[:, pcx],
            y=components[:, pcy],
            color=color,
            height=height,
            width=width,
            color_discrete_sequence=color_discrete_sequence,
        )
        .update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        .update_layout(
            title=title,
            xaxis_title=f"PC{pcx + 1}",
            yaxis_title=f"PC{pcy + 1}",
        )
    )

    for i, feature in enumerate(pca_columns):
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=loadings[i, pcx],
            y1=loadings[i, pcy],
            line=dict(width=4),
        )
        fig.add_annotation(
            x=loadings[i, pcx],
            y=loadings[i, pcy],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    return fig


def plot_model(
    X,
    color,
    x_comp=0,
    y_comp=1,
    width=None,
    height=None,
    title=None,
    axis_title_root: str = "PC",
    loadings=None,
    column_names=None,
    marginal_x=None,
    marginal_y=None,
    hover_data=None,
    rescale: bool = True,
):

    if rescale is True:
        x = X[:, x_comp] * (1.0 / (X[:, x_comp].max() - X[:, x_comp].min()))
        y = X[:, y_comp] * (1.0 / (X[:, y_comp].max() - X[:, y_comp].min()))
    else:
        x = X[:, x_comp]
        y = X[:, y_comp]
    df = pd.DataFrame({"x": x, "y": y, "color": color, "hover_data": hover_data})

    fig = px.scatter(
        data_frame=df,
        x="x",
        y="y",
        color="color",
        height=height,
        width=width,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        hover_data=["hover_data"],
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )

    x_padding = (np.absolute(x.max()) + np.abs(x.min())) / 20
    y_padding = (np.absolute(y.max()) + np.abs(y.min())) / 20

    fig.update_xaxes(range=[x.min() - x_padding, x.max() + x_padding])
    fig.update_yaxes(range=[y.min() - y_padding, y.max() + y_padding])
    fig.update_layout(
        title=title,
        xaxis_title=f"{axis_title_root}{x_comp + 1}",
        yaxis_title=f"{axis_title_root}{y_comp + 1}",
        legend_title="OIV value",
    )

    if loadings is not None:
        loadings[:, x_comp] = loadings[:, x_comp] * (
            1.0 / (loadings[:, x_comp].max() - loadings[:, x_comp].min())
        )
        loadings[:, y_comp] = loadings[:, y_comp] * (
            1.0 / (loadings[:, y_comp].max() - loadings[:, y_comp].min())
        )
        xc, yc = [], []
        for i in range(loadings.shape[0]):
            xc.extend([0, loadings[i, x_comp], None])
            yc.extend([0, loadings[i, y_comp], None])
        fig.add_trace(
            go.Scatter(
                x=xc,
                y=yc,
                mode="lines",
                name="Loadings",
                showlegend=False,
                line=dict(color="black"),
                opacity=0.3,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=loadings[:, x_comp],
                y=loadings[:, y_comp],
                mode="text",
                text=column_names,
                opacity=0.7,
                name="Loadings",
            ),
        )

    return fig


def observations_sankey(clean_steps, width=1600, height=600):
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=[
                        "raw merge",  # 0
                        "kept after clean raw merge",  # 1
                        "discarded after clean raw merge",  # 2
                        "kept after numeric dataframe",  # 3
                        "discarded after numeric dataframe",  # 4
                        "kept after inverted numeric dataframe",  # 5
                        "discarded after inverted numeric dataframe",  # 6
                    ],
                    x=[0.1, 0.33, 0.33, 0.75, 0.70, 0.99, 0.85],
                    y=[0.1, 0.1, 0.8, 0.5, 0.7, 0.1, 0.3],
                ),
                link=dict(
                    source=[0, 0, 1, 1, 1, 1],
                    target=[1, 2, 3, 4, 5, 6],
                    value=[
                        clean_steps["clean_raw_merge"][0],
                        clean_steps["clean_raw_merge"][1],
                        clean_steps["numeric_dataframe"][0],
                        clean_steps["numeric_dataframe"][1],
                        clean_steps["inverted_numeric_dataframe"][0],
                        clean_steps["inverted_numeric_dataframe"][1],
                    ],
                    color=[
                        "steelblue",
                        "red",
                        "forestgreen",
                        "indianred",
                        "forestgreen",
                        "indianred",
                    ],
                ),
                arrangement="snap",
            )
        ]
    )
    fig.update_layout(width=width, height=height)

    return fig


def plot_balance_histogram(labels, color, is_text, width=None, height=None):
    fig = px.histogram(
        x=labels,
        color=color,
        text_auto=is_text,
        width=width,
        height=height,
    )
    return fig
