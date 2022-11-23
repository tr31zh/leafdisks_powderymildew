import os
import warnings
import itertools

import pandas as pd

import streamlit as st
from streamlit_yellowbrick import st_yellowbrick

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from yellowbrick.cluster import (
    KElbowVisualizer,
    SilhouetteVisualizer,
    InterclusterDistance,
)
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import gav_mildiou_func as gof
import gav_mildiou_const as goc
import gav_mildiou_text as got
import gav_mildiou_plot_plotly as gop

pd.options.plotting.backend = "plotly"
pd.options.display.float_format = "{:4,.2f}".format

warnings.simplefilter("ignore")

st.set_page_config(
    page_title="GAV Mildiou Data Wrangling",
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
    return gop.plot_variance(df_ev=df_ev)


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
    return gof.filter_csvs()


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


st.markdown(
    """
<style>
.streamlit-expanderHeader {
    font-size: 2em;
    font-weight: bolder;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(got.txt_title)


col_target, col_explain = st.columns(2)

with col_target:
    st.markdown(got.txt_target)

with col_explain:
    st.markdown(got.txt_libraries)

col_about_streamlit, col_expand_collapse_all = st.columns([3, 2])
with col_about_streamlit:
    st.markdown(got.txt_about_streamlit)

with col_expand_collapse_all:
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("  ")
    st.markdown("  ")
    expand_all = st.checkbox(label="Expand/collapse all sections", value=True)

with st.expander(got.txt_lvl2_header_wio, expanded=expand_all):
    col_desc_oiv, col_desc_variables = st.columns(2)

    with col_desc_oiv:
        st.markdown(f"{goc.lvl_3_header} OIV")
        st.markdown(got.txt_oiv_452_spec)
        st.image(os.path.join(goc.datain_path, "images", "OIV_examples.png"))
        st.warning(
            "OIV 452-2 is a resistance scale, higher note means less disease phenotype"
        )

    with col_desc_variables:
        st.markdown(f"{goc.lvl_3_header} Other variables")
        st.markdown("Other variables with which we want to predict OIV 452-2")
        st.image(os.path.join(goc.datain_path, "images", "oiv_452-1_desc.png"))
        st.markdown(got.txt_oiv_452_spec_header)

    # st.write(gop.plot_sample_oiv_images(), use_container_width=True)
    st.plotly_chart(
        gop.plot_sample_oiv_images(width=1000, height=900), use_container_width=True
    )

    st.markdown("{goc.lvl_3_header} The aim of this dashboard")
    st.markdown(got.txt_what_we_want)


with st.expander(got.txt_lvl2_header_build_database, expanded=expand_all):
    lcl_csv_files = get_local_csvs()

    st.markdown(f"{goc.lvl_3_header} Retrieve distant Excels")
    st.markdown(got.txt_get_excels)

    if os.path.isfile(goc.path_to_df_result) is False:
        files = get_distant_excels()

        st.write(files)

        st.markdown("Copying files")
        current_pg = st.progress(0)

        copy_excel_files(files.path.to_list())

        current_pg.progress(1.0)
    else:
        st.markdown("Excel files already parsed")
    st.success("")

    st.markdown(f"{goc.lvl_3_header} Build CSVs")

    st.markdown(got.txt_excel_headers)

    st.markdown("Building CSVs")

    current_pg = st.progress(0)

    df_result = filter_csvs()

    current_pg.progress(1.0)

    lcl_csv_files = [
        os.path.join(goc.mildiou_extracted_csvs_path, filename)
        for filename in df_result.csv_file_name.dropna().to_list()
    ]

    st.markdown("Extracted CSVs")
    st.write(lcl_csv_files)

    sample_csv = st.selectbox(
        label="Select a CSV to view:",
        options=lcl_csv_files,
        index=0,
        format_func=lambda x: os.path.basename(x),
    )

    col_sample_df, col_sample_describe = st.columns(2)
    df_smpl = pd.read_csv(sample_csv)
    with col_sample_df:
        st.markdown("Selected dataframe")
        print_dataframe_and_shape(df_smpl)

    with col_sample_describe:
        st.markdown("Dataframe description, Na values are ignored")
        st.write(df_smpl.drop(["colonne"], axis=1).describe())

    col_rej_csv_text, col_rej_csv_hist = st.columns([2, 3])

    with col_rej_csv_text:
        st.markdown(got.txt_rejected_csvs)
        df_corrupted = gof.sheet_filtering_out_df(df_result)

        df_corrupted.to_csv(
            os.path.join(goc.datain_path, "corrupted_excels.csv"),
            index=False,
            sep=";",
        )

        st.markdown("**Which ones are corrupted?**")
        st.dataframe(df_corrupted)

    with col_rej_csv_hist:
        st.plotly_chart(
            gop.plot_rejected_hist(df_result),
            use_container_width=True,
        )

    st.info("Info sheets have no data, only experiment descriptors")

    clean_steps = {}

    st.success("")

    st.markdown(f"{goc.lvl_3_header} Merge CSVs")
    st.markdown("Load all CSVs into one dataframe and show the first 100 rows")
    df_raw_merged = build_raw_merged(lcl_csv_files)
    st.dataframe(df_raw_merged.head(n=100))
    st.markdown(df_raw_merged.shape)

    clean_steps["raw_merge"] = (df_raw_merged.shape[0], 0)

    col_data_inconsistency_spec, col_data_inconsistency_plot = st.columns(2)
    with col_data_inconsistency_spec:
        st.markdown(got.txt_oiv_452_spec_cs)
        st.warning("Inconsistent rows will be deleted")

    with col_data_inconsistency_plot:
        st.plotly_chart(
            gop.plot_inconsistencies(
                df_raw_merged,
                sort_values=False,
                width=goc.two_plot_width + 100,
                height=goc.two_plot_height,
            )
        )

    st.markdown(f"{goc.lvl_5_header} Sheets with inconsistent data")
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
        st.markdown("Total amount of inconsistencies types")
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

    col_data_consistency_spec, col_data_consistency_plot = st.columns(2)

    df_merged = clean_merged_dataframe(df_raw_merged)
    clean_steps["clean_raw_merge"] = (
        df_merged.shape[0],
        df_raw_merged.shape[0] - df_merged.shape[0],
    )

    with col_data_consistency_spec:
        st.markdown(f"{goc.lvl_5_header} Consistent dataframe")
        st.markdown("New dataframe")
        print_dataframe_and_shape(df_merged)
        st.info(
            f"After removing inconsistent lines we went from {df_raw_merged.shape[0]} to {df_merged.shape[0]} consistent rows"
        )

    with col_data_consistency_plot:
        st.plotly_chart(
            gop.plot_inconsistencies(
                df_merged,
                width=goc.two_plot_width + 100,
                height=goc.two_plot_height,
                title="No visible errors remain in the data",
            )
        )

    st.success("")

with st.expander(got.txt_lvl2_header_data_overview, expanded=expand_all):
    col_balance, col_nans = st.columns([3, 2])

    with col_balance:
        st.markdown(f"{goc.lvl_3_header} Set balance")
        st.plotly_chart(
            gop.plot_balance_histogram(
                labels=df_merged.oiv.sort_values().astype(str),
                color=df_merged.oiv.sort_values().astype(str),
                is_text=True,
                width=1000,
                height=600,
            ),
            use_container_width=True,
        )
        st.markdown("Set balance looks good")

    with col_nans:
        st.markdown(f"{goc.lvl_3_header} NaN values")
        st.write({c: df_merged[c].isna().sum() for c in df_merged.columns})
        st.markdown(
            """
            **NaN values happen when**:
            - If "necrosis" is 0, "surface_necrosee" and "taille_necrose" are NaN
            - If "sporulation" is 0, "densite_sporulation" is NaN
            """
        )

    st.markdown(f"{goc.lvl_3_header} Numeric dataframe")

    st.markdown(
        """
    We remove all columns that are not linked to the regression and drop all rows with **NaN** values as they **will not be accepted by the models**
    """
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

    clean_steps["numeric_dataframe"] = (
        df_num.shape[0],
        df_merged.shape[0] - df_num.shape[0],
    )

    col_df_num, col_df_num_balance, col_explain = st.columns(3)

    with col_df_num:
        st.markdown(f"{goc.lvl_4_header} Heat map")
        st.plotly_chart(
            px.imshow(
                df_num.sort_values(["oiv", "necrose", "sporulation"]).reset_index(
                    drop=True
                ),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
            ),
            use_container_width=True,
        )

    with col_df_num_balance:
        st.markdown(f"{goc.lvl_4_header} New set balance")
        st.plotly_chart(
            gop.plot_balance_histogram(
                labels=df_num.oiv.sort_values().astype(str),
                color=df_num.oiv.sort_values().astype(str),
                is_text=True,
                width=goc.three_plot_width,
                height=goc.three_plot_height,
            ),
            use_container_width=True,
        )

    with col_explain:
        st.markdown(f"{goc.lvl_4_header} Result")
        st.markdown(" ")
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


with st.expander(got.txt_lvl2_header_invert_axes, expanded=expand_all):
    st.markdown(got.txt_fail)

    col_inv_df, con_inv_num_df = st.columns([3, 2])

    df_inverted = cache_invert_axis(df_merged)

    df_inverted = df_inverted[
        [df_inverted.columns[i] for i in [9, 8, 4, 6, 0, 3, 2, 10, 5, 7, 1]]
    ].sort_values(["oiv", "sporulation", "necrose"])

    df_inv_num = (
        df_inverted.drop(["colonne"], axis=1)
        .select_dtypes(exclude=object)
        .drop_duplicates()
    )

    clean_steps["inverted_numeric_dataframe"] = (
        df_inv_num.shape[0],
        df_merged.shape[0] - df_inv_num.shape[0],
    )

    with col_inv_df:
        st.markdown(f"{goc.lvl_3_header} The head of new dataframe")
        st.dataframe(df_inverted.head(10))
        st.markdown(df_inverted.shape)

    with con_inv_num_df:
        st.markdown(f"{goc.lvl_3_header} The numeric dataframe")
        st.dataframe(df_inv_num)
        st.markdown(df_inv_num.shape)

    df_inv_num = df_inv_num.sort_values(
        [
            "oiv",
            "necrose",
            "taille_necrose",
            "surface_necrosee",
            "sporulation",
            "densite_sporulation",
        ]
    )

    st.markdown(f"{goc.lvl_3_header} Some plots")

    st.markdown(f"{goc.lvl_4_header} Evolution of available rows")
    st.plotly_chart(
        gop.observations_sankey(clean_steps=clean_steps),
        use_container_width=True,
    )
    st.markdown(
        f"""
    We started with {df_raw_merged.shape[0]} annotations and after removing inconsistent data, 
    columns that are not needed and duplicates we end up with {df_inv_num.shape[0]} observations
    """
    )

    col_inv_df_num, col_inv_df_num_balance, col_inv_explain = st.columns(3)

    with col_inv_df_num:
        st.markdown(f"{goc.lvl_4_header} Heat map")
        st.plotly_chart(
            px.imshow(
                df_inv_num.sort_values(["oiv", "necrose", "sporulation"]).reset_index(
                    drop=True
                ),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
            ),
            use_container_width=True,
        )

    with col_inv_df_num_balance:
        st.markdown(f"{goc.lvl_4_header} New set balance")
        st.plotly_chart(
            px.histogram(
                x=df_inv_num.oiv.sort_values().astype(str),
                color=df_inv_num.oiv.sort_values().astype(str),
                text_auto=True,
                width=goc.three_plot_width,
                height=goc.three_plot_height,
            ),
            use_container_width=True,
        )
        st.markdown("We still have a somewhat well balanced set")

    with col_inv_explain:
        st.markdown(f"{goc.lvl_4_header} Result")
        st.markdown(" ")
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

    with inv_vio_plot:
        st.markdown(f"{goc.lvl_4_header} Violin plot")
        fig = make_subplots(rows=1, cols=len(df_inv_num.columns))
        for i, var in enumerate(df_inv_num.columns):
            fig.add_trace(
                go.Violin(y=df_inv_num[var], name=var),
                row=1,
                col=i + 1,
            )
        fig.update_traces(points="all", jitter=0.3).update_layout(
            height=goc.three_plot_height,
            width=goc.three_plot_width * 2,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with inv_corr_plot:
        st.markdown(f"{goc.lvl_4_header} Correlation matrix")
        st.plotly_chart(
            px.imshow(
                df_inv_num.drop_duplicates().corr(),
                text_auto=True,
                height=goc.three_plot_height,
                width=goc.three_plot_width,
            ),
            use_container_width=True,
        )
        st.markdown("OIV is not as correlated as expected with the other variables")

    st.markdown(f"{goc.lvl_3_header} OIV homogeneity")

    col_oiv_1, col_oiv_3, col_oiv_5, col_oiv_homo_txt = st.columns(4)

    with col_oiv_1:
        st.plotly_chart(
            gop.plot_oiv_homogeneity(
                df_src=df_inv_num,
                oiv=1,
                width=goc.four_plot_width,
                height=goc.four_plot_height,
            ),
            use_container_width=True,
        )

    with col_oiv_3:
        st.plotly_chart(
            gop.plot_oiv_homogeneity(
                df_src=df_inv_num,
                oiv=3,
                width=goc.four_plot_width,
                height=goc.four_plot_height,
            ),
            use_container_width=True,
        )

    with col_oiv_5:
        st.plotly_chart(
            gop.plot_oiv_homogeneity(
                df_src=df_inv_num,
                oiv=5,
                width=goc.four_plot_width,
                height=goc.four_plot_height,
            ),
            use_container_width=True,
        )

    with col_oiv_homo_txt:
        st.markdown(got.txt_homogenity_txt)

    col_oiv_7, col_oiv_9, col_oiv_avg, col_oiv_avg_txt = st.columns(4)

    with col_oiv_7:
        st.plotly_chart(
            gop.plot_oiv_homogeneity(
                df_src=df_inv_num,
                oiv=7,
                width=goc.four_plot_width,
                height=goc.four_plot_height,
            ),
            use_container_width=True,
        )

    with col_oiv_9:
        st.plotly_chart(
            gop.plot_oiv_homogeneity(
                df_src=df_inv_num,
                oiv=9,
                width=goc.four_plot_width,
                height=goc.four_plot_height,
            ),
            use_container_width=True,
        )
    with col_oiv_avg:
        st.plotly_chart(
            gop.plot_avg_by_oiv(
                df_inv_num,
                height=goc.four_plot_height,
                width=goc.four_plot_width,
            ),
            use_container_width=True,
        )

    with col_oiv_avg_txt:
        st.markdown(got.txt_homogenity_avg_txt)

    st.markdown(f"{goc.lvl_3_header} Models")

    Xi = df_inv_num
    yi = df_inv_num.oiv.astype(int)
    Xi = Xi.drop(["oiv"], axis=1)
    scaler = StandardScaler()
    scaler.fit(Xi)
    Xi = scaler.transform(Xi)

    Xi.shape

    inv_pca_txt, inv_plsda_txt, inv_lda_txt = st.columns(3)
    inv_pca, inv_plsda, inv_lda = st.columns(3)
    inv_x, inv_y, _ = st.columns(3)

    with inv_x:
        inv_x_comp = st.selectbox(
            label="Inverted principal component for x axis, applies to PCA and PLS-DA",
            options=[i + 1 for i in range(Xi.shape[1] - 1)],
            index=0,
        )

    with inv_y:
        inv_y_comp = st.selectbox(
            label="Inverted principal component for y axis, applies to PCA and PLS-DA",
            options=[i + 1 for i in range(Xi.shape[1] - 1)],
            index=1,
        )

    with inv_pca_txt:
        st.markdown(got.txt_model_def_pca)

    with inv_plsda_txt:
        st.markdown(got.txt_model_def_plsda)

    with inv_lda_txt:
        st.markdown(got.txt_model_def_lda)

    with inv_pca:

        st.plotly_chart(
            gop.plot_model(
                X=PCA().fit_transform(Xi),
                x_comp=inv_x_comp - 1,
                y_comp=inv_y_comp - 1,
                color=yi.astype(str),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
                title="Inverted PCA 2D",
            ),
            use_container_width=True,
        )

    with inv_plsda:
        pls_data_all_inv = PLSRegression(n_components=Xi.shape[1])
        x_new = pls_data_all_inv.fit(Xi, yi).transform(Xi)
        st.plotly_chart(
            gop.plot_model(
                X=x_new,
                x_comp=inv_x_comp - 1,
                y_comp=inv_y_comp - 1,
                color=yi.astype(str),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
                title=f"Inverted PLS-DA, score: {pls_data_all_inv.score(Xi, yi)}",
                axis_title_root="X-variate ",
            ),
            use_container_width=True,
        )

    with inv_lda:
        lda_data_all_inv = LinearDiscriminantAnalysis()
        x_new = lda_data_all_inv.fit(Xi, yi).transform(Xi)
        st.plotly_chart(
            gop.plot_model(
                X=x_new,
                x_comp=0,
                y_comp=1,
                color=yi.astype(str),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
                title=f"Inverted LDA score: {lda_data_all_inv.score(Xi, yi)}",
                axis_title_root="X-variate ",
            ),
            use_container_width=True,
        )

    st.markdown(f"{goc.lvl_3_header} Check overlapping")

    st.markdown(
        "Some observations seem to overlap, we're going to check that one point in the vectorial space codes only one OIV"
    )

    d = cache_build_dup_df(df_inverted)

    df_dup = d["df_dup"]
    pairs = d["pairs"]
    qtty = d["count"]

    col_dup_df, col_dup_unique, col_dup_count = st.columns([3, 1, 1])

    with col_dup_df:
        st.markdown(f"{goc.lvl_4_header} What unique rows can code as OIV")
        print_dataframe_and_shape(df_dup)

    with col_dup_unique:
        st.markdown(f"{goc.lvl_4_header} Truly unique rows")
        print_dataframe_and_shape(
            df_inv_num.drop(["oiv"], axis=1).drop_duplicates().reset_index(drop=True)
        )

    with col_dup_count:
        st.markdown(f"{goc.lvl_4_header} Duplicate count")
        print_dataframe_and_shape(pd.DataFrame(data={"pair": pairs, "count": qtty}))

    st.markdown(f"{goc.lvl_4_header} Sheet by sheet unique rows can code as OIV")
    col_sbs_oiv, col_sbs_txt = st.columns([5, 2])
    with col_sbs_oiv:
        print_dataframe_and_shape(
            cache_build_sbs_dup_df(df_inverted)
            .sort_values(["experiment", "sheet"])
            .reset_index(drop=True)
            .set_index(["experiment", "sheet"])
        )
    with col_sbs_txt:
        st.warning("Inconsistency with OIV and the vaiables")
        st.markdown(got.txt_sbs_dup_txt)

    st.markdown(f"{goc.lvl_3_header} Sheet by sheet prediction")

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
            ),
            use_container_width=True,
        )
    st.markdown(got.txt_duprate_vs_prediction)

    st.markdown(f"{goc.lvl_3_header} Single sheet prediction")

    sbs_sng_sel, sbs_sng_dups, sbs_sng_scatter = st.columns([1, 3, 3])

    with sbs_sng_sel:
        st.markdown(f"{goc.lvl_4_header} Sheet selection")
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
        st.markdown(f"{goc.lvl_4_header} Sheet")
        print_dataframe_and_shape(df_es)
        st.markdown(f"{goc.lvl_4_header} Duplicate predictions")
        print_dataframe_and_shape(gof.build_dup_df(df_es)["df_dup"])

    with sbs_sng_scatter:
        st.markdown(f"{goc.lvl_4_header} PLS-DA")
        X = df_es.drop(["oiv"], axis=1)
        y = df_es.oiv
        X = StandardScaler().fit(X).transform(X)
        es_pls_da = PLSRegression(n_components=X.shape[1]).fit(X, y)
        st.plotly_chart(
            gop.plot_model(
                X=es_pls_da.x_scores_,
                x_comp=1 - 1,
                y_comp=2 - 1,
                color=y.astype(str),
                height=goc.two_plot_height,
                width=goc.two_plot_width,
                title=f"PLS-DA for experiment {exp} sheet {sheet}score: {es_pls_da.score(X, y)}",
                axis_title_root="X-variate ",
            ),
            use_container_width=True,
        )

    st.markdown(got.txt_rem_nec_spo)

    col_sel_oiv, col_sel_col, col_sel_target = st.columns([1, 2, 1])

    with col_sel_oiv:
        oiv_classes = st.multiselect(
            label="Select IOV classes in model",
            options=goc.odd_numbers,
            default=goc.odd_numbers,
        )

    with col_sel_col:
        acbo = list(df_inv_num.columns)
        acbo.remove("oiv")
        selected_columns = st.multiselect(
            label="Select columns in model",
            options=acbo,
            default=acbo,
        )

    df_inv_num_wosn = (
        df_inv_num[df_inv_num.oiv.isin(oiv_classes)][selected_columns + ["oiv"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    yi_wond = df_inv_num_wosn.oiv
    X_wond = df_inv_num_wosn.drop(["oiv"], axis=1)
    scaler = StandardScaler()
    scaler.fit(X_wond)
    X_wond = scaler.transform(X_wond)

    inv_pca, inv_plsda = st.columns([1, 1])

    with inv_pca:
        st.markdown(f"{goc.lvl_4_header} PCA")
        st.plotly_chart(
            gop.plot_model(
                X=PCA().fit_transform(X_wond),
                x_comp=0,
                y_comp=1,
                color=yi_wond.astype(str),
                width=goc.two_plot_width,
                height=goc.two_plot_height,
                title="Inverted PCA 2D",
            ),
            use_container_width=True,
        )

    with inv_plsda:
        st.markdown(f"{goc.lvl_4_header} PLS-DA")
        pls_data_all_inv = PLSRegression(n_components=X_wond.shape[1])
        x_new = pls_data_all_inv.fit(X_wond, yi_wond).transform(X_wond)

        st.plotly_chart(
            gop.plot_model(
                X=pls_data_all_inv.x_scores_,
                x_comp=0,
                y_comp=1,
                color=yi_wond.astype(str),
                width=goc.two_plot_width,
                height=goc.two_plot_height,
                title=f"Inverted PLS-DA, score: {pls_data_all_inv.score(X_wond, yi_wond)}",
                axis_title_root="X-variate ",
            ),
            use_container_width=True,
        )

with st.expander(got.txt_lvl2_header_kmeans, expanded=expand_all):
    st.markdown(got.txt_kmeans)

    X_km = df_inv_num.drop(["oiv"], axis=1).drop_duplicates().reset_index(drop=True)

    st.markdown(got.txt_noiv_sel_cut)

    colt_cut_plot, col_cut_text = st.columns([3, 1])
    with colt_cut_plot:
        fig = make_subplots(rows=2, cols=3)

        for (r, c), sort_order in zip(
            itertools.product([1, 2, 3], [1, 2, 3]),
            itertools.permutations(
                ["taille_necrose", "surface_necrosee", "densite_sporulation"]
            ),
        ):
            fig.add_trace(
                go.Heatmap(
                    z=X_km.drop(["sporulation", "necrose"], axis=1)
                    .sort_values(list(sort_order))
                    .drop_duplicates()
                    .reset_index(drop=True),
                    x=sort_order,
                ),
                row=r,
                col=c,
            )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=sort_order,
            ),
            width=goc.three_plot_width * 2,
            height=goc.two_plot_height - 50,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_cut_text:
        st.markdown(got.txt_noiv_sel_cut_outcome)

    st.markdown(got.txt_kmeans_pca)

    xkm_pca = PCA()
    x_pca = xkm_pca.fit_transform(X_km)

    col_kmeans_pca, col_kmeans_variance, col_kmeans_loadings = st.columns(3)

    with col_kmeans_pca:
        fig = px.scatter_3d(
            x=x_pca[:, 0],
            y=x_pca[:, 1],
            z=x_pca[:, 2],
            title="PCA",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col_kmeans_variance:
        st.plotly_chart(
            gop.plot_variance(
                df_ev=pd.DataFrame.from_dict(
                    {
                        "pc": [
                            f"PC{i}"
                            for i in range(len(xkm_pca.explained_variance_ratio_))
                        ],
                        "exp_var_per": xkm_pca.explained_variance_ratio_ * 100,
                    }
                ),
                width=goc.three_plot_width,
                height=goc.three_plot_height,
            ),
            use_container_width=True,
        )

    with col_kmeans_loadings:
        df_loadings: pd.DataFrame = pd.DataFrame(
            xkm_pca.components_.T * xkm_pca.explained_variance_ratio_,
            columns=[f"PC{i+1}" for i in range(len(xkm_pca.components_))],
            index=X_km.columns,
        )
        fig = df_loadings.T.plot.bar()
        fig.update_layout(
            width=goc.three_plot_width,
            height=goc.three_plot_height,
            title="Loadings",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown("It appears that **3** components are enough")
    xkm_pca = PCA(n_components=3)
    x_pca = xkm_pca.fit_transform(X_km)

    st.markdown(got.txt_kmeans_explore_cluster_count)
    col_km = [st.columns(3), st.columns(3), st.columns(3)]
    col_km = list(itertools.chain(*col_km))

    for i, col in enumerate(col_km):
        with col:
            nc = i + 2
            st.markdown(f"{goc.lvl_5_header} {nc} classes")
            km = KMeans(n_clusters=nc, init="k-means++", random_state=42)
            y_km = km.fit_predict(x_pca).astype(int)
            fig = px.scatter_3d(
                data_frame=pd.DataFrame(
                    {
                        "x": x_pca[:, 0],
                        "y": x_pca[:, 1],
                        "z": x_pca[:, 2],
                        "color": y_km.astype(str),
                    }
                ).sort_values(["color"]),
                x="x",
                y="y",
                z="z",
                width=goc.three_plot_width,
                height=goc.three_plot_height,
                color="color",
            )
            fig.add_trace(
                go.Scatter3d(
                    x=km.cluster_centers_[:, 0],
                    y=km.cluster_centers_[:, 1],
                    z=km.cluster_centers_[:, 2],
                    name="",
                    mode="markers",
                    marker=go.Marker(symbol="x", size=6, color="black"),
                    showlegend=False,
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.markdown(
        """
    Visually, it seems that the best class count is 3.
    """
    )

    col_elbow_txt, col_elbow_plot = st.columns(2)
    col_silhouette_txt, col_silhouette_plot = st.columns(2)

    with col_elbow_txt:
        st.markdown(got.txt_kmeans_elbow)

    with col_elbow_plot:
        elbow_model = KMeans(init="k-means++", random_state=42)
        elb_visualizer = KElbowVisualizer(elbow_model, k=(2, 13))
        elb_visualizer.fit(x_pca)
        st_yellowbrick(elb_visualizer)

    st.markdown(got.txt_kmeans_silhouette)

    yb_width = 500
    yb_height = 500

    col_sil = [st.columns(3), st.columns(3), st.columns(3)]
    col_sil = list(itertools.chain(*col_sil))

    for i, col in enumerate(col_sil):
        with col:
            nc = i + 2
            st.markdown(f"{goc.lvl_5_header} {nc} classes")
            silhouette_model = KMeans(init="k-means++", n_clusters=nc, random_state=42)
            sil_visualizer = SilhouetteVisualizer(
                silhouette_model, size=(yb_width, yb_height)
            )
            sil_visualizer.fit(x_pca)
            st_yellowbrick(sil_visualizer)

    st.markdown(got.txt_kmeans_what)

    col_icd = st.columns(3)
    for nc, col in zip([3, 6, 8], col_icd):
        with col:
            st.markdown(f"{goc.lvl_5_header} {nc} classes")
            icd_model = KMeans(init="k-means++", n_clusters=nc, random_state=42)
            icd_visualizer = InterclusterDistance(icd_model, size=(yb_width, yb_height))
            icd_visualizer.fit(x_pca)
            st_yellowbrick(icd_visualizer)

    st.markdown(got.txt_noiv_select_oiv)

    cols_hm = [st.columns(3), st.columns(3), st.columns(3)]
    cols_hm = list(itertools.chain(*cols_hm))

    for i, col in enumerate(cols_hm):
        with col:
            nc = i + 2
            st.markdown(f"{goc.lvl_5_header} Data heatmap for {nc} classes")
            df_hm = (
                X_km.assign(
                    noiv=KMeans(n_clusters=nc, init="k-means++", random_state=42)
                    .fit_predict(x_pca)
                    .astype(int)
                )
                .drop(["sporulation", "necrose"], axis=1)
                .drop_duplicates()
                .sort_values(
                    [
                        "noiv",
                        "taille_necrose",
                        "surface_necrosee",
                        "densite_sporulation",
                    ]
                )
                .reset_index(drop=True)
            )[
                [
                    "noiv",
                    "taille_necrose",
                    "surface_necrosee",
                    "densite_sporulation",
                ]
            ]
            st.plotly_chart(
                px.imshow(
                    df_hm,
                    width=goc.three_plot_width,
                    height=goc.three_plot_height,
                ),
                use_container_width=True,
            )
    st.markdown(got.txt_km_hm_conclusion)

with st.expander(got.txt_lvl2_header_conclusion, expanded=expand_all):
    st.markdown(got.txt_conclusion)
