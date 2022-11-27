from cgitb import html
import datetime
import json
from pathlib import Path

from PIL import Image

import numpy as np
import pandas as pd
import pandas_dash

import plotly.express as px
from dash import Dash, Input, State, Output, callback, dash_table, dcc, html, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import gav_mildiou_const as goc

import ld_dataset as ldd
import ld_th_pl_lightning as ldpl
import ld_image as ldi

all_columns = [
    "experiment",
    "rep",
    "image_name",
    "ligne",
    "colonne",
    "oiv",
    "sporulation",
    "densite_sporulation",
    "necrose",
    "surface_necrosee",
    "taille_necrose",
]

var_columns = [
    "image_name",
    "ligne",
    "colonne",
    "oiv",
    "sporulation",
    "densite_sporulation",
    "necrose",
    "surface_necrosee",
    "taille_necrose",
]


path_to_here = Path(__file__).parent
checkpoint_path = path_to_here.parent.joinpath(
    "notebooks",
    "lightning_logs",
    "version_12",
    "checkpoints",
    "epoch=34-step=105-train_loss=0.0088628-val_loss=0.01398.ckpt",
)
annotations_path = path_to_here.parent.joinpath("data_out", "annotations")

LOCAL_IMAGES = True

csv_path = path_to_here.parent.joinpath(
    goc.datain_path,
    "local_raw_merged.csv" if LOCAL_IMAGES is True else "raw_merged.csv",
)

index_columns = ["experiment", "rep", "image_name", "ligne", "colonne"]

predictor = ldpl.LeafDiskSegmentationPredictor(
    model=ldpl.LeafDiskSegmentation.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path)
    )
)

df = (
    pd.read_csv(str(csv_path), sep=";")
    .assign(rep=lambda x: x.experiment.str.split(pat="_", expand=True)[1])
    .assign(experiment=lambda x: x.experiment.str.split(pat="_", expand=True)[0])
    .assign(rep=lambda x: x.rep.str.replace("saisie", "NA"))[all_columns]
    .drop_duplicates()
    .sort_values(["experiment", "rep", "image_name", "ligne", "colonne"])
)


def read_user_data(user_name) -> dict:
    try:
        jf = open(annotations_path.joinpath(f"{user_name}.json"), "r")
    except:
        return None
    else:
        return json.load(jf)


def write_user_data(user_name, data) -> None:
    with open(
        annotations_path.joinpath(f"{user_name}.json"), "w", encoding="utf-8"
    ) as jf:
        json.dump(data, jf)


def get_user_names() -> list:
    return [f.stem for f in annotations_path.glob("*.json")]


def image_to_plot(image, width, height):
    fig = px.imshow(image, width=width, height=height)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def get_local_path_from_name(image_name) -> str:
    ret = ldd.una_images_folder.joinpath(image_name + ".jpg")
    if ret.is_file() is True:
        return ret
    ret = ldd.train_images_folder.joinpath(image_name + ".jpg")
    if ret.is_file() is True:
        return ret
    ret = ldd.wot_images_folder.joinpath(image_name + ".jpg")
    if ret.is_file() is True:
        return ret

    return None


flat_data, flat_columns = df[index_columns].dash.to_dash_table()

tab_user = dbc.Tab(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.Row(
                        dbc.Col(html.H4("User", className="card-title")),
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Col(
                            cb_user_names := dcc.Dropdown(
                                options=get_user_names(),
                                id="cb_user_names",
                            ),
                            width=4,
                        ),
                        align="center",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                input_add_user := dbc.Input(
                                    id="input_add_user",
                                    placeholder="New user name",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                bt_add_user := dbc.Button("Add user", id="bt_add_user")
                            ),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Col(html.H4("Projects", className="card-title")),
                        align="center",
                        style={"margin-top": "20px"},
                    ),
                    dbc.Row(
                        dbc.Col(
                            cb_user_projects := dcc.Dropdown(
                                options=[],
                                id="cb_user_projects",
                            ),
                            width=4,
                        ),
                        align="center",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                input_add_project := dbc.Input(
                                    id="input_add_project",
                                    placeholder="New project name",
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                bt_add_project := dbc.Button(
                                    "Add project", id="bt_add_project"
                                )
                            ),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Col(html.H4("Select variables", className="card-title")),
                        align="center",
                        style={"margin-top": "20px"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                rd_select_variables := dbc.RadioItems(
                                    options=[
                                        {"label": "OIV only", "value": 1},
                                        {
                                            "label": "OIV and standard variables",
                                            "value": 2,
                                        },
                                        {"label": "OIV and new variables", "value": 3},
                                        {"label": "Custom variables", "value": 4},
                                    ],
                                    value=2,
                                    id="radioitems-inline-input",
                                    inline=True,
                                    labelCheckedClassName="text-success",
                                    inputCheckedClassName="border border-success bg-success",
                                )
                            ),
                            dbc.Col(
                                input_var_count := dbc.Input(
                                    type="number",
                                    value=2,
                                    min=1,
                                    max=10,
                                    step=1,
                                    placeholder="Variable count for custom variables",
                                ),
                                width=3,
                            ),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        dbc.Col(
                            variables_holder := html.Div(
                                id="variables_holder",
                                children=[],
                            )
                        ),
                        align="center",
                        style={"margin-top": "20px"},
                    ),
                ],
                className="pad-row",
            )
        ),
    ],
    id="tab_user",
    tab_id="tab_user",
    label="User & project settings",
    activeTabClassName="fw-bold fst-italic",
    activeLabelClassName="text-success",
)

tab_df_filter = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(dbc.Label("Rows per page"), width="auto"),
                        dbc.Col(
                            cb_row_count := dcc.Dropdown(
                                options=[10, 25, 50, 100],
                                value=25,
                                clearable=False,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            chk_show_annotations := dbc.Switch(
                                id="standalone-switch",
                                label="Show annotations",
                                value=False,
                            )
                        ),
                    ],
                    justify="between",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                cb_experiment := dcc.Dropdown(
                                    [x for x in sorted(df.experiment.unique())],
                                    placeholder="Experiment",
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                cb_rep := dcc.Dropdown(
                                    [x for x in sorted(df.rep.unique())],
                                    placeholder="Repetitiion",
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                cb_oiv := dcc.Dropdown(
                                    [x for x in sorted(df.oiv.unique())],
                                    placeholder="OIV",
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                input_rnd_count := dbc.Input(
                                    type="number",
                                    min=0,
                                    max=1000,
                                    step=20,
                                    placeholder="Random selection count",
                                )
                            ]
                        ),
                    ],
                    justify="between",
                    className="mt-3 mb-4",
                ),
                ld_df := dash_table.DataTable(
                    data=flat_data,
                    columns=flat_columns,
                    id="table-leaf-disks",
                    page_size=25,
                    sort_action="native",
                    sort_mode="single",
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {"if": {"column_id": c}, "textAlign": "left"}
                        for c in ["country", "iso_alpha3"]
                    ],
                ),
            ],
        ),
        className="mt-3",
    ),
    id="tab_filtering",
    tab_id="tab_filtering",
    label="Dataframe filtering:",
    activeTabClassName="fw-bold fst-italic",
    activeLabelClassName="text-success",
)

tab_df_updated = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                df_updated := dash_table.DataTable(
                    data=None,
                    columns=None,
                    id="df_updated",
                    page_size=25,
                    sort_action="native",
                    sort_mode="single",
                    filter_action="native",
                    style_cell_conditional=[  # align text columns to left. By default they are aligned to right
                        {"if": {"column_id": c}, "textAlign": "left"}
                        for c in ["country", "iso_alpha3"]
                    ],
                ),
            ],
        ),
        className="mt-3",
    ),
    id="tab_df_updated",
    tab_id="tab_df_updated",
    label="Updated dataframe",
    activeTabClassName="fw-bold fst-italic",
    activeLabelClassName="text-success",
)

tab_ld_overview = dbc.Tab(
    dbc.Card(
        dbc.CardBody(
            [
                currrent_store := dcc.Store(id="current_index"),
                dummy_store := dcc.Store(id="dummy_store"),
                dbc.Row(
                    [
                        dbc.Col(
                            div_image := html.Div(
                                children=[], id="div_image", style={"max-width": "100%"}
                            ),
                            width=8,
                        ),
                        dbc.Col(
                            [
                                annotations_holder := html.Div(
                                    id="annotations_holder",
                                    children=[],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            bt_save_annotation := dbc.Button(
                                                "Save annotation",
                                                id="bt_save_annotation",
                                            ),
                                            width="auto",
                                        ),
                                        dbc.Col(
                                            bt_next_annotation := dbc.Button(
                                                "Next disc",
                                                color="warning",
                                                id="bt_next_annotation",
                                            ),
                                            width="auto",
                                        ),
                                    ],
                                    justify="between",
                                ),
                                dbc.Row(
                                    collapse_src_annotations := dbc.Accordion(
                                        [
                                            dbc.AccordionItem(
                                                [
                                                    div_src_annotations := html.Div(
                                                        id="div_src_annotations"
                                                    ),
                                                ],
                                                title="Show Excel annotations",
                                                item_id="div_src_annotations",
                                            ),
                                        ],
                                        start_collapsed=True,
                                    ),
                                    style={"margin-top": "15px"},
                                ),
                            ],
                            width=4,
                            className="p-3 border bg-light",
                        ),
                    ]
                ),
            ]
        )
    ),
    id="tab_overview",
    tab_id="tab_overview",
    label="Leaf disk overview",
    activeTabClassName="fw-bold fst-italic",
    activeLabelClassName="text-success",
)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(st_user := dbc.Alert(), align="start"),
                dbc.Col(st_project := dbc.Alert(), align="start"),
                dbc.Col(
                    pg_progress := dbc.Progress(style={"height": "30px"}),
                    align="center",
                ),
                dbc.Col(error_message := html.Div([]), align="end"),
            ]
        ),
        tabs := dbc.Tabs(
            [
                tab_user,
                tab_df_filter,
                tab_ld_overview,
                tab_df_updated,
            ],
            id="tabs",
        ),
    ]
)


def filter_with_dict(df: pd.DataFrame(), d: dict) -> pd.DataFrame():
    tmp = df.copy()
    for k, v in d.items():
        tmp = tmp[tmp[k] == v]
    return tmp


@callback(
    # Outputs
    Output(cb_user_names, "options"),
    Output(cb_user_names, "value"),
    Output(cb_user_projects, "options"),
    Output(cb_user_projects, "value"),
    Output(input_add_user, "value"),
    Output(input_add_project, "value"),
    Output(tab_df_filter, "disabled"),
    # Inputs
    Input(bt_add_user, "n_clicks"),
    Input(bt_add_project, "n_clicks"),
    Input(cb_user_names, "value"),
    Input(cb_user_projects, "value"),
    # States
    State(input_add_user, "value"),
    State(cb_user_names, "options"),
    State(input_add_project, "value"),
    State(cb_user_projects, "options"),
)
def update_user_and_project(
    nc_add_user,
    nc_add_projedct,
    sel_user,
    sel_project,
    new_user,
    user_names,
    new_project,
    project_names,
):
    users = user_names
    user = sel_user
    projects = project_names
    project = sel_project

    caller_id = ctx.triggered_id if not None else "noone"

    if caller_id == "bt_add_user":
        if new_user:
            if new_user in user_names:
                users, user = user_names, new_user
            else:
                write_user_data(
                    user_name=new_user,
                    data={
                        "user_name": new_user,
                        "created": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                        "last_viewed": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                        "projects": {},
                    },
                )
                user_names.append(new_user)
                users, user = user_names, new_user
        else:
            users, user = user_names, None
    elif caller_id == "cb_user_names":
        user = sel_user
    elif caller_id == "bt_add_project":
        if new_project and user:
            if new_project in project_names:
                projects, project = project_names, new_project
            else:
                data = read_user_data(user_name=user)
                data["last_viewed"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                data["projects"][new_project] = {"empty": True}
                write_user_data(user_name=user, data=data)
                project = new_project
    elif caller_id == "cb_user_projects":
        project = sel_project
    else:
        pass
    try:
        projects = list(read_user_data(user_name=user)["projects"].keys())
    except:
        projects = []

    return (
        users,
        user,
        projects,
        project,
        "",
        "",
        (
            user
            and project
            and (
                "empty"
                not in read_user_data(user_name=user)["projects"][project].keys()
            )
        ),
    )


@callback(
    Output(df_updated, "data"),
    Output(df_updated, "columns"),
    Input(bt_save_annotation, "n_clicks"),
    State(currrent_store, "data"),
    State(cb_user_names, "value"),
    State({"type": "annotation_var_value", "index": ALL}, "value"),
    State({"type": "annotation_var_name", "index": ALL}, "children"),
    State(cb_user_projects, "value"),
)
def save_annotation(
    _,
    currrent_store,
    current_user,
    annotation_var_value,
    annotation_var_name,
    projects_name,
):
    caller_id = ctx.triggered_id if not None else "noone"
    if caller_id != "bt_save_annotation":
        raise PreventUpdate
    if current_user and currrent_store:
        try:
            data = read_user_data(user_name=current_user)
            df_tmp = pd.read_json(data["projects"][projects_name]["dataframe"])
        except OSError:
            raise PreventUpdate
        if df_tmp is not None:
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            filter = (
                (df.image_name == currrent_store["image_name"])
                & (df.ligne == currrent_store["ligne"])
                & (df.colonne == currrent_store["colonne"])
            )
            for name, value in zip(annotation_var_name, annotation_var_value):
                df_tmp.loc[filter, f"vn:{name}"] = value
            df_tmp.loc[filter, "annotation_timestamp"] = ts
            df_tmp.loc[filter, "annotated"] = True

            data["projects"][projects_name]["dataframe"] = df_tmp.to_json()
            data["last_viewed"] = ts

            write_user_data(user_name=current_user, data=data)

            return df_tmp.dash.to_dash_table()
        else:
            return None, None
    else:
        return None, None


@callback(
    Output(div_src_annotations, "children"),
    Input(currrent_store, "data"),
    Input(collapse_src_annotations, "active_item"),
    State(cb_user_names, "value"),
    State(cb_user_projects, "value"),
)
def show_annotations(currrent_store, active_item, current_user, projects_name):
    if active_item is None or "div_src_annotations" not in active_item:
        return None
    data = read_user_data(user_name=current_user)
    print(currrent_store)
    if data is not None:
        df_tmp = pd.read_json(data["projects"][projects_name]["dataframe"])
        item = filter_with_dict(df=df_tmp, d=currrent_store)

        table_header = [html.Thead(html.Tr([html.Th("Variable"), html.Th("Value")]))]
        rows = []
        for col in var_columns:
            rows.append(html.Tr([html.Td(col), html.Td(item[col].to_list()[0])]))
        table_body = [html.Tbody([row for row in rows])]
        return dbc.Table(table_header + table_body, bordered=True)
    else:
        return None


@callback(
    Output(div_image, "children"),
    Output(currrent_store, "data"),
    Output({"type": "annotation_var_value", "index": ALL}, "value"),
    Output(pg_progress, "value"),
    Input(bt_next_annotation, "n_clicks"),
    State(cb_user_names, "value"),
    State({"type": "annotation_var_value", "index": ALL}, "value"),
    State(cb_user_projects, "value"),
)
def next_image(n_click, current_user, annotation_var_value, projects_name):
    caller_id = ctx.triggered_id if not None else "noone"
    if caller_id != "bt_next_annotation":
        raise PreventUpdate
    data = read_user_data(user_name=current_user)
    if data is not None:
        df_tmp = pd.read_json(data["projects"][projects_name]["dataframe"])
        item = df_tmp[df_tmp.annotated == False].sample(n=1)
        image_path = get_local_path_from_name(item.image_name.to_list()[0])
        row = item.ligne.to_list()[0]
        col = item.colonne.to_list()[0]
        try:
            print(image_path)
            img = ldd.open_image(str(image_path))
        except OSError:
            raise PreventUpdate

        predicted_mask = predictor.predict_image(image_path)
        clean_mask = ldi.clean_contours(mask=predicted_mask.copy(), size_thrshold=0.75)
        contours = ldi.index_contours(clean_mask)

        img_all = image_to_plot(
            ldi.print_single_contour(
                clean_mask,
                contours,
                row=row,
                col=col,
                canvas=img.copy(),
                rect_thickness=40,
            ),
            width=150,
            height=100,
        )

        img_ld = image_to_plot(
            ldi.get_leaf_disk(
                img.copy(),
                contours,
                row,
                col,
                padding=10,
            ),
            width=800,
            height=800,
        )

        step, total = df_tmp[df_tmp.annotated == True].shape[0], df_tmp.shape[0]

        return (
            [
                dbc.Row(dcc.Graph(figure=img_ld, config={"fillFrame": True})),
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                f"Image: {image_path.stem} - [{row}, {col}] --- {step} of {total} completed"
                            )
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=img_all,
                                config={
                                    "staticPlot": True,
                                    "displayModeBar": False,
                                    "displaylogo": False,
                                    "editable": False,
                                    "fillFrame": True,
                                },
                            ),
                            width={"size": 3, "order": 5},
                            align="start",
                        ),
                    ],
                ),
            ],
            {
                "image_name": item.image_name.to_list()[0],
                "ligne": item.ligne.to_list()[0],
                "colonne": item.colonne.to_list()[0],
            },
            [None for _ in annotation_var_value],
            step,
        )
    else:
        return (
            dbc.Alert("Please wait...", color="primary"),
            {},
            [None for _ in annotation_var_value],
            0,
        )


@callback(
    Output(variables_holder, "children"),
    Input(rd_select_variables, "value"),
    Input(input_var_count, "value"),
)
def update_variable_list(rb_value, input_var_count):
    if rb_value == 1:
        return [
            dbc.Input(
                id={"type": "variable_name", "index": 1}, value="OIV", disabled=True
            )
        ]
    elif rb_value == 2:
        return [
            dbc.Input(
                id={"type": "variable_name", "index": i + 1},
                value=var_name,
                disabled=True,
            )
            for i, var_name in zip(
                range(6),
                [
                    "oiv",
                    "sporulation",
                    "densite_sporulation",
                    "necrose",
                    "surface_necrosee",
                    "taille_necrose",
                ],
            )
        ]
    elif rb_value == 3:
        return [
            dbc.Input(
                id={"type": "variable_name", "index": i + 1},
                value=var_name,
                disabled=True,
            )
            for i, var_name in zip(
                range(6),
                [
                    "oiv",
                    "taux de sporulation",
                    "taux de necrose noire",
                    "taux de necrose marron",
                    "taux de sénescence",
                ],
            )
        ]
    elif rb_value == 4:
        return [
            dbc.Input(
                id={"type": "variable_name", "index": i + 1},
                placeholder="Enter variable name",
            )
            for i in range(input_var_count)
        ]
    else:
        return []


@callback(
    Output(st_user, "children"),
    Output(st_user, "color"),
    Output(st_project, "children"),
    Output(st_project, "color"),
    Input(cb_user_names, "value"),
    Input(cb_user_projects, "value"),
)
def update_status(user, project):
    return (
        f"User: {user}" if user else "Please select or create an user",
        "success" if user else "warning",
        f"Project: {project}" if project else "Please select or create a project",
        "success" if project else "warning",
    )


@callback(Output(ld_df, "page_size"), Input(cb_row_count, "value"))
def update_row_count(row_count):
    return int(row_count)


def filter_dataframe(exp, rep, oiv, rnd_count):
    df_tmp = df.copy()
    if exp is not None:
        df_tmp = df_tmp[df_tmp.experiment == exp]
        if rep is not None:
            df_tmp = df_tmp[df_tmp.rep == rep]
        if oiv is not None:
            df_tmp = df_tmp[df_tmp.oiv == oiv]
    else:
        df_tmp = df_tmp.copy()

    if isinstance(rnd_count, int) and rnd_count > 0:
        df_tmp = df_tmp.sample(n=rnd_count)

    return df_tmp


@callback(
    Output("tabs", "active_tab"),
    Output(error_message, "children"),
    Output(annotations_holder, "children"),
    Output(pg_progress, "min"),
    Output(pg_progress, "max"),
    Input("tabs", "active_tab"),
    State(cb_user_names, "value"),
    State(cb_user_projects, "value"),
    State(cb_experiment, "value"),
    State(cb_rep, "value"),
    State(cb_oiv, "value"),
    State(input_rnd_count, "value"),
    State({"type": "variable_name", "index": ALL}, "value"),
)
def on_switch_tab(
    new_tab,
    user_name,
    project_name,
    experiment,
    rep,
    oiv,
    rnd_count,
    var_names,
):
    if new_tab == "tab_overview":
        df_tmp = filter_dataframe(
            exp=experiment,
            rep=rep,
            oiv=oiv,
            rnd_count=rnd_count,
        )

        if not user_name:
            return (
                "tab_user",
                dbc.Alert(
                    "Please select an user.",
                    color="danger",
                    id="alert_no_user",
                ),
                [],
                0,
                0,
            )

        if not project_name:
            return (
                "tab_user",
                dbc.Alert(
                    "Please select a project.",
                    color="danger",
                    id="alert_no_user",
                ),
                [],
                0,
                0,
            )

        data = read_user_data(user_name=user_name)
        data["last_viewed"] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if data["projects"][project_name].pop("empty", False) == True:

            df_tmp = df_tmp.assign(
                annotateur=user_name,
                annotated=False,
                annotation_timestamp=None,
                xor=False,
                year_folder=lambda x: "EXP-20" + x.experiment.str.slice(3, 5, 1),
            ).assign(**{f"vn:{var_name}": None for var_name in var_names})
            data["projects"][project_name]["row_count"] = df_tmp.shape[0]
            data["projects"][project_name]["dataframe"] = df_tmp.to_json()
            data["projects"][project_name]["variables"] = var_names
            write_user_data(user_name=user_name, data=data)

        data = read_user_data(user_name=user_name)
        variables = data["projects"][project_name]["variables"]
        row_count = data["projects"][project_name]["row_count"]

        return (
            "tab_overview",
            [],
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(
                            var_name,
                            id={"type": "annotation_var_name", "index": i + 1},
                        ),
                        dbc.Input(
                            id={"type": "annotation_var_value", "index": i + 1},
                            placeholder=var_name,
                            type="number",
                            min=1,
                            max=9,
                        ),
                    ],
                    className="mb-3",
                )
                for i, var_name in enumerate(variables)
            ],
            0,
            row_count,
        )
    else:
        return new_tab, [], [], 0, 0


@callback(
    Output(ld_df, "data"),
    Output(ld_df, "columns"),
    Output(tab_df_filter, "label"),
    Input(chk_show_annotations, "value"),
    Input(cb_experiment, "value"),
    Input(cb_rep, "value"),
    Input(cb_oiv, "value"),
    Input(input_rnd_count, "value"),
)
def on_filter_dataframe(is_show_annotations, experiment, rep, oiv, rnd_count):
    df_tmp = filter_dataframe(
        exp=experiment,
        rep=rep,
        oiv=oiv,
        rnd_count=rnd_count,
    )

    data, columns = (
        df_tmp[all_columns]
        if is_show_annotations is True
        else df_tmp[index_columns].reset_index(drop=True)
    ).dash.to_dash_table()

    return (
        data,
        columns,
        f"Dataframe filtering: {len(data)} items in list",
    )


@callback(Output(cb_rep, "options"), Input(cb_experiment, "value"))
def update_repetitions(experiment):
    return df[df.experiment == experiment].rep.unique()


@callback(
    Output(cb_oiv, "options"),
    Input(cb_experiment, "value"),
    Input(cb_rep, "value"),
)
def update_repetitions(experiment, rep):
    if experiment is not None:
        df_tmp = df[df.experiment == experiment]
        if rep is not None:
            df_tmp = df_tmp[df_tmp.rep == rep]
        return df_tmp.oiv.unique()
    else:
        return []


if __name__ == "__main__":
    app.run_server(debug=True)
