import os
import warnings

import pandas as pd

import streamlit as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import gav_oidium_func as gof
import gav_oidium_const as goc
import gav_oidium_text as got
import gav_oidium_plot_plotly as gop

pd.options.plotting.backend = "plotly"
pd.options.display.float_format = "{:4,.2f}".format

warnings.simplefilter("ignore")

st.set_page_config(
    page_title="Extaedio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def st_progress(step, total):
    current_pg((step + 1) / total)


def print_dataframe_and_shape(df):
    st.dataframe(df)
    st.markdown(df.shape)


@st.cache
def plot_variance(df_ev):
    return gof.plot_variance(df_ev=df_ev)


def get_oiv_cat(df):
    return gof.get_oiv_cat(df)


@st.cache
def get_common_columns(csv_files):
    return gof.get_common_columns(csv_files=csv_files)


@st.cache()
def build_inconsistencies_dataframe(df_source):
    return gof.build_inconsistencies_dataframe(df_source=df_source)


@st.cache()
def clean_merged_dataframe(df_source):
    return gof.clean_merged_dataframe(df_source=df_source)


@st.cache
def get_distant_excels():
    return gof.get_distant_excels()


@st.cache(suppress_st_warning=True)
def copy_excel_files(files):
    gof.copy_excel_files(files, st_progress)


@st.cache
def filter_csvs():
    return gof.filter_csvs(st_progress)


@st.cache()
def get_local_csvs():
    return gof.get_local_csvs()


@st.cache()
def build_raw_merged(lcl_csv_files):
    return gof.build_raw_merged(lcl_csv_files=lcl_csv_files)


@st.cache()
def build_sbs_plsda(df_src, df_dup):
    return gof.build_sbs_plsda(df_src=df_src, df_dup=df_dup)


@st.cache()
def cache_build_dup_df(df):
    return gof.build_dup_df(df)


@st.cache()
def cache_build_sbs_dup_df(df_src):
    return gof.build_sbs_dup_df(df_src)


@st.cache()
def cache_invert_axis(df_src):
    return gof.invert_axis(df_src=df_src)


st.markdown("# Leaf Disk Collate Ground Truth")

col_target, col_explain = st.columns(2)

with col_target:
    st.markdown(got.txt_target)

with col_explain:

    st.markdown(got.txt_target)

st.markdown("## What is OIV and how do we want to predict it")

col_desc_oiv, col_desc_variables = st.columns(2)

with col_desc_oiv:
    st.markdown("### OIV")
    st.markdown(got.txt_oiv_452_spec)
    st.image(os.path.join(goc.datain_path, "images", "OIV_examples.png"))
    st.warning(
        "OIV 452-2 is a resistance scale, higher note means less disease phenotype"
    )

with col_desc_variables:
    st.markdown("### Other variables")
    st.markdown("Other variables with which we want to predict OIV 452-2")
    st.image(os.path.join(goc.datain_path, "images", "oiv_452-1_desc.png"))
    st.markdown(got.txt_oiv_452_spec_req)


st.markdown("### The aim of this dashboard")
st.markdown(
    "Can we predict the OIV from the other variables, on other words is there a link between the variables ond the OIV?"
)

st.markdown("## Build dataframe")


lcl_csv_files = get_local_csvs()

st.markdown("### Retrieve distant Excels")
st.markdown(got.txt_get_excels)

if os.path.isfile(goc.path_to_df_result) is False:
    files = get_distant_excels()

    st.write(files)

    st.write("Copying files")
    current_pg = st.progress(0)

    copy_excel_files(files)

    current_pg.progress(1.0)
else:
    st.write("Excel files already parsed")
st.success("")

st.markdown("### Build CSVs")

st.markdown(got.txt_excel_headers)

st.write("Building CSVs")

current_pg = st.progress(0)

df_result = filter_csvs()

current_pg.progress(1.0)

lcl_csv_files = [
    os.path.join(goc.oidium_extracted_csvs_path, filename)
    for filename in df_result.csv_file_name.dropna().to_list()
]

st.write("Extracted CSVs")
st.write(lcl_csv_files)

st.write(f"A sample file: {lcl_csv_files[13]}")
sample_csv = st.selectbox(
    label="Select a CSV to view:",
    options=lcl_csv_files,
    index=0,
    format_func=lambda x: os.path.basename(x),
)

col_sample_df, col_sample_describe = st.columns(2)
df_smpl = pd.read_csv(sample_csv)
with col_sample_df:
    print_dataframe_and_shape(df_smpl)

with col_sample_describe:
    st.write(df_smpl.drop(["colonne"], axis=1).describe())

col_rej_csv_text, col_rej_csv_hist = st.columns([1, 3])

with col_rej_csv_text:
    st.markdown(got.txt_rejected_csvs)

with col_rej_csv_hist:
    st.plotly_chart(gop.plot_rejected_hist(df_result))

st.markdown("#### Which ones are corrupted")

df_corrupted = (
    df_result[
        df_result.comment.isin(
            [
                "Corrupted dataframe",
                "Corrupted dataframe, failed to retrieve photos",
            ]
        )
    ]
    .drop(["csv_file_name", "outcome"], axis=1)
    .reset_index(drop=True)
)

df_corrupted.to_csv(
    os.path.join(goc.datain_path, "corrupted_excels.csv"),
    index=False,
    sep=";",
)

st.dataframe(df_corrupted)
st.info("Info sheets have no data, only experiment descriptors")

st.success("")

st.markdown("### Merge CSVs")
st.markdown("Load all CSVs into one dataframe and show the first 100 rows")
df_raw_merged = build_raw_merged(lcl_csv_files)
st.dataframe(df_raw_merged.head(n=100))
st.markdown(df_raw_merged.shape)

st.write("#### Data consistency check")
st.markdown(got.txt_oiv_452_spec_req)
st.plotly_chart(gof.plot_inconsistencies(df_raw_merged, sort_values=False))
st.warning("Various rows are inconsistent")
st.markdown("After removing inconsistent lines we get a new consistent dataframe")
df_merged = clean_merged_dataframe(df_raw_merged)
st.dataframe(df_merged.head(50))
st.markdown(df_merged.shape)
st.plotly_chart(gof.plot_inconsistencies(df_merged))
st.info(f"We went from {df_raw_merged.shape[0]} to {df_merged.shape[0]} consistent rows")
st.write("List of sheets with inconsistent data")
df_inconsistent = build_inconsistencies_dataframe(df_raw_merged)

col_inc_data, col_inc_sum = st.columns(2)

with col_inc_data:
    st.markdown(
        """
    - **oob**: Out of bounds, value outside of permitted values
    - **n_inc**: Linked values inconsistent

    """
    )

    st.dataframe(df_inconsistent)


with col_inc_sum:
    st.write("Total amount of inconsistencies types")
    cols = [
        "sporulation_oob",
        "sporulation_ds_inc",
        "densite_sporulation_oob",
        "necrose_oob",
        "necrose_sn_inc",
        "necrose_tn_inc",
        "taille_necrose_oob",
        "surface_necrosee_oob",
        "oiv_oob",
        "oiv_s_inc",
        "ligne_oob",
    ]
    st.table(
        pd.DataFrame(
            data={"sum": [df_inconsistent[col].sum() for col in cols]},
            index=cols,
        )
    )

st.success("")

st.markdown("## Data overview")

col_balance, col_nans = st.columns([3, 2])

with col_balance:
    st.markdown("### Set balance")

    st.plotly_chart(
        px.histogram(
            x=df_merged.oiv.sort_values().astype(str),
            color=df_merged.oiv.sort_values().astype(str),
            text_auto=True,
            width=1000,
            height=600,
        )
    )

with col_nans:
    st.markdown("### NaN values")
    st.write({c: df_merged[c].isna().sum() for c in df_merged.columns})
    st.write(
        """
        **NaN values happen when**:
        - If "necrosis" is 0, "surface_necrosee" and "taille_necrose" are NaN
        - If "sporulation" is 0, "densite_sporulation" is NaN
        """
    )

st.markdown("### Numeric dataframe")

st.write(
    "We remove all columns that are not linked to the regression and drop all rows with **NaN** values as they **will not be accepted by the models**"
)

df_num = (
    df_merged.drop(["colonne"], axis=1)
    .dropna()
    .select_dtypes(exclude=object)
    .drop_duplicates()
)
df_num_cols = df_num.columns
df_num_cols = [
    df_num_cols[3],
    df_num_cols[0],
    df_num_cols[2],
    df_num_cols[4],
    df_num_cols[1],
    df_num_cols[5],
]
df_num = df_num[df_num_cols].sort_values(["oiv", "sporulation", "necrose"])

col_df_num, col_df_num_balance, col_explain = st.columns(3)

with col_df_num:
    st.markdown("#### Heat map")
    st.plotly_chart(
        px.imshow(
            df_num.sort_values(["oiv", "necrose", "sporulation"]).reset_index(drop=True),
            width=500,
            height=400,
        )
    )

with col_df_num_balance:
    st.markdown("#### New set balance")
    st.plotly_chart(
        px.histogram(
            x=df_num.oiv.sort_values().astype(str),
            color=df_num.oiv.sort_values().astype(str),
            text_auto=True,
            width=500,
            height=400,
        )
    )

with col_explain:
    st.markdown("#### Result")
    st.write(" ")
    st.info(f"There are only {df_num.shape[0]} observations left")
    st.markdown(
        """
        Two hypothesis:
        - There are only this amount of phenotypes possible
        - The human eye can only discriminate this many
        """
    )
    st.error(
        "Since OIV 9 implies no sporulation, there are no longer rows with OIV value 9"
    )

st.info(
    "There are no rows with OIV 9, this makes building a model pointless, we'll find another way"
)


st.markdown("## Inverting the axes")

st.markdown(got.txt_fail)

col_inv_df, con_inv_num_df = st.columns([3, 2])


df_inverted = cache_invert_axis(df_merged)

df_inverted = df_inverted[
    [df_inverted.columns[i] for i in [9, 8, 4, 6, 0, 3, 2, 10, 5, 7, 1]]
].sort_values(["oiv", "sporulation", "necrose"])


df_inv_num = (
    df_inverted.drop(["colonne"], axis=1).select_dtypes(exclude=object).drop_duplicates()
)


with col_inv_df:
    st.markdown("### The head of new dataframe")
    st.dataframe(df_inverted.head(10))
    st.markdown(df_inverted.shape)

with con_inv_num_df:
    st.markdown("### The head of numeric dataframe")
    st.dataframe(df_inv_num.head(10))
    st.markdown(df_inv_num.shape)


df_inv_num = df_inv_num.sort_values(["oiv", "sporulation", "necrose"])

st.markdown("### Some plots")
col_inv_df_num, col_inv_df_num_balance, col_inv_explain = st.columns(3)

with col_inv_df_num:
    st.markdown("#### Heat map")
    st.plotly_chart(
        px.imshow(
            df_inv_num.sort_values(["oiv", "necrose", "sporulation"]).reset_index(
                drop=True
            ),
            width=500,
            height=400,
        )
    )

with col_inv_df_num_balance:
    st.markdown("#### New set balance")
    st.plotly_chart(
        px.histogram(
            x=df_inv_num.oiv.sort_values().astype(str),
            color=df_inv_num.oiv.sort_values().astype(str),
            text_auto=True,
            width=500,
            height=400,
        )
    )

with col_inv_explain:
    st.markdown("#### Result")
    st.write(" ")
    st.info(
        f"There are {df_inv_num.shape[0]} observations left instead of the previous {df_num.shape[0]}"
    )
    st.markdown(
        """
        Two hypothesis:
        - There are only this amount of phenotypes possible
        - The human eye can only discriminate this many
        """
    )


inv_vio_plot, inv_corr_plot = st.columns([2, 1])

inv_plots_width = 600
inv_plots_height = 400
with inv_vio_plot:
    st.markdown("#### Violin plot")
    fig = make_subplots(rows=1, cols=len(df_inv_num.columns))
    for i, var in enumerate(df_inv_num.columns):
        fig.add_trace(
            go.Violin(y=df_inv_num[var], name=var),
            row=1,
            col=i + 1,
        )
    fig.update_traces(points="all", jitter=0.3).update_layout(
        height=inv_plots_height,
        width=inv_plots_width * 2,
    )
    st.plotly_chart(fig)

with inv_corr_plot:
    st.markdown("#### Correlation matrix")
    st.plotly_chart(
        px.imshow(
            df_inv_num.drop_duplicates().corr(),
            text_auto=True,
            height=inv_plots_height,
            width=inv_plots_width,
        )
    )

st.markdown("### OIV homogeneity")

col_oiv_1, col_oiv_3, col_oiv_5 = st.columns(3)
col_oiv_width = 500
col_oiv_height = 500

with col_oiv_1:
    st.plotly_chart(
        gop.plot_oiv_homogeneity(
            df_src=df_inv_num,
            oiv=1,
            width=col_oiv_width,
            height=col_oiv_height,
        )
    )

with col_oiv_3:
    st.plotly_chart(
        gop.plot_oiv_homogeneity(
            df_src=df_inv_num,
            oiv=3,
            width=col_oiv_width,
            height=col_oiv_height,
        )
    )

with col_oiv_5:
    st.plotly_chart(
        gop.plot_oiv_homogeneity(
            df_src=df_inv_num,
            oiv=5,
            width=col_oiv_width,
            height=col_oiv_height,
        )
    )

col_oiv_7, col_oiv_9, col_oiv_avg = st.columns(3)

with col_oiv_7:
    st.plotly_chart(
        gop.plot_oiv_homogeneity(
            df_src=df_inv_num,
            oiv=7,
            width=col_oiv_width,
            height=col_oiv_height,
        )
    )

with col_oiv_9:
    st.plotly_chart(
        gop.plot_oiv_homogeneity(
            df_src=df_inv_num,
            oiv=9,
            width=col_oiv_width,
            height=col_oiv_height,
        )
    )
with col_oiv_avg:
    st.plotly_chart(
        gop.plot_avg_by_oiv(df_inv_num, height=col_oiv_height, width=col_oiv_width)
    )

st.markdown("### Models")

Xi = df_inv_num
yi = df_inv_num.oiv
Xi = Xi.drop(["oiv"], axis=1)
scaler = StandardScaler()
scaler.fit(Xi)
Xi = scaler.transform(Xi)

Xi.shape


_, inv_x, inv_y, _ = st.columns([1, 1, 1, 1])

with inv_x:
    inv_x_comp = st.selectbox(
        label="Inverted principal component for x axis",
        options=[i for i in range(Xi.shape[1] - 1)],
        index=0,
    )

with inv_y:
    inv_y_comp = st.selectbox(
        label="Inverted principal component for y axis",
        options=[i for i in range(Xi.shape[1] - 1)],
        index=1,
    )

inv_pca, inv_splsda = st.columns([1, 1])

with inv_pca:
    st.markdown("#### PCA")
    st.plotly_chart(
        gop.plot_model(
            X=PCA().fit_transform(Xi),
            x_comp=inv_x_comp,
            y_comp=inv_y_comp,
            color=yi.astype(str),
            width=800,
            height=700,
            title="Inverted PCA 2D",
        )
    )

with inv_splsda:
    st.markdown("#### sPLSDA")
    pls_data_all_inv = PLSRegression(n_components=Xi.shape[1])
    x_new = pls_data_all_inv.fit(Xi, yi).transform(Xi)

    st.plotly_chart(
        gop.plot_model(
            X=pls_data_all_inv.x_scores_,
            x_comp=inv_x_comp,
            y_comp=inv_y_comp,
            color=yi.astype(str),
            width=800,
            height=700,
            title="Inverted sPLS-DA",
        )
    )
    st.markdown(f"**sPLSDA score**: {pls_data_all_inv.score(Xi, yi)}")

st.markdown("### Check overlapping")

st.write(
    "Some observations seem to overlap, we're going to check that one point in the vectorial space codes only one OIV"
)

d = cache_build_dup_df(df_inverted)

df_dup = d["df_dup"]
pairs = d["pairs"]
qtty = d["count"]

col_dup_df, col_dup_unique, col_dup_count = st.columns([3, 1, 1])

with col_dup_df:
    st.markdown("#### What unique rows can code as OIV")
    print_dataframe_and_shape(df_dup)


with col_dup_unique:
    st.markdown("#### Truly unique rows")
    print_dataframe_and_shape(
        df_inv_num.drop(["oiv"], axis=1).drop_duplicates().reset_index(drop=True)
    )

with col_dup_count:
    st.markdown("#### Duplicate count")
    print_dataframe_and_shape(pd.DataFrame(data={"pair": pairs, "count": qtty}))

st.markdown("#### Sheet by sheet unique rows can code as OIV")
print_dataframe_and_shape(
    cache_build_sbs_dup_df(df_inverted)
    .sort_values(["experiment", "sheet"])
    .reset_index(drop=True)
    .set_index(["experiment", "sheet"])
)


st.markdown("### Sheet by sheet prediction")

st.markdown(
    f"The prediction is bad at {pls_data_all_inv.score(Xi, yi)}, we try next to predict sheet by sheet to see the results"
)

df_sheet_plsda = build_sbs_plsda(df_inverted, df_dup)

sbs_col, sbs_plot = st.columns(2)

with sbs_col:
    df_sheet_plsda = df_sheet_plsda.sort_values(
        [
            "row_count",
            "score",
            "experiment",
            "sheet",
        ],
        ascending=False,
    ).reset_index(drop=True)

    st.dataframe(df_sheet_plsda)

with sbs_plot:
    st.plotly_chart(
        px.scatter(
            data_frame=df_sheet_plsda[
                ((df_sheet_plsda.score > -1) & (df_sheet_plsda.score <= 1))
            ].assign(row_count=lambda x: x.row_count.astype(float)),
            y="score",
            x="dup_rate",
            color="row_count",
            color_continuous_scale=px.colors.sequential.OrRd,
            trendline="ols",
            trendline_color_override="blue",
        )
    )

st.markdown("### Single sheet prediction")

sbs_sng_sel, sbs_sng_dups, sbs_sng_scatter = st.columns([1, 3, 3])

with sbs_sng_sel:
    st.markdown("#### Sheet selection")
    exp = st.selectbox(
        label="Experiement",
        options=list(df_inverted.experiment.sort_values(ascending=True).unique()),
        index=0,
    )
    if exp != "Select one":
        sheet = st.selectbox(
            label="Sheet",
            options=list(df_inverted[df_inverted.experiment == exp].sheet.unique()),
            index=0,
        )

df_es = (
    df_inverted[((df_inverted.experiment == exp) & (df_inverted.sheet == sheet))]
    .select_dtypes(exclude=object)
    .drop(["colonne"], axis=1)
    .drop_duplicates()
)

with sbs_sng_dups:
    st.markdown("#### Duplicate predictions")
    print_dataframe_and_shape(gof.build_dup_df(df_es)["df_dup"])

with sbs_sng_scatter:
    st.markdown("#### sPLS-DA")
    X = df_es.drop(["oiv"], axis=1)
    y = df_es.oiv
    X = StandardScaler().fit(X).transform(X)
    es_pls_da = PLSRegression(n_components=X.shape[1]).fit(X, y)
    st.plotly_chart(
        gop.plot_model(
            X=es_pls_da.x_scores_,
            x_comp=1,
            y_comp=2,
            color=y.astype(str),
            height=700,
            width=700,
            title=f"sPLS-DA for experiment {exp} sheet {sheet}score: {es_pls_da.score(X, y)}",
        )
    )
