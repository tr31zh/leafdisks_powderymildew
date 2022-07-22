import os
import shutil
from pathlib import Path
import itertools
from matplotlib.pyplot import title
from functools import reduce

from tqdm import tqdm

import pandas as pd
import numpy as np

import warnings

warnings.simplefilter("ignore")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression, CCA, PLSSVD
from skimage import io as skio

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

pd.options.plotting.backend = "plotly"
pd.options.display.float_format = "{:4,.2f}".format


import streamlit as st

st.set_page_config(
    page_title="Extaedio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

datain_path = os.path.join(".", "data_in")
excel_file_path = os.path.join(datain_path, "oidium_source_excels", "")
oidium_extracted_csvs_path = os.path.join(datain_path, "oidium_extracted_csvs", "")
excel_file_list_path = os.path.join(excel_file_path, "excel_list.txt")
path_to_df_result = os.path.join(datain_path, "extracted_csv_files.csv")
odd_numbers = [1, 3, 5, 7, 9]
available_templates = list(pio.templates.keys())

needed_columns = ["nomphoto", "oiv", "s", "sq", "n", "fn", "tn", "ligne", "colonne"]

oiv_452_spec_req = """
**From the specifications we now that a clean dataframe has the following rules**:
- _sporulation_ **must be** 1 ou 0
- if _sporulation_ **is** 0 , _densite_sporulation_ **must be** NaN else it **must be** an odd number
- _densite_sporulation_ **must be** a number and **not** 0
- _necrosis_ **must be** 1 ou 0
- if _necrosis_ **is** 1 _surface_necrosee_ & _taille_necrose_ **must not be** none else they **must**
- _surface_necrosee_ & _taille_necrose_ **must be** NaN or odd
- _OIV_ **must be** an odd number
- if _OIV_ is 9 **there must be no** _sporulation_ else **there must be**
- _ligne_ **must not** be NA
"""


def check_list_in_list(required_columns, available_columns):
    failures = []
    for rc in required_columns:
        if rc not in available_columns:
            failures.append(rc)

    return True if len(failures) == 0 else failures


def print_dataframe_and_shape(df):
    st.dataframe(df)
    st.markdown(df.shape)


def plot_variance(df_ev):
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
        height=700,
        width=800,
        title="Explained variance by different principal components",
        xaxis_title="Principal component",
        yaxis_title="Explained variance in percent",
    )
    return ev_fig


def plot_inconsistencies(df, sort_values: bool = True):
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
        height=1000,
        width=1400,
        xaxis_title="Value",
        yaxis_title="Count",
    )

    return fig


def get_oiv_cat(df):
    return df.oiv.astype(str)


@st.cache
def get_common_columns(csv_files):
    common_columns = set(pd.read_csv(csv_files[0]).columns.to_list())
    columns_occ = {}
    for filepath in csv_files:
        cu_columns = pd.read_csv(filepath).columns.to_list()
        for c in cu_columns:
            if c in columns_occ:
                columns_occ[c] += 1
            else:
                columns_occ[c] = 1
        common_columns = common_columns.intersection(set(cu_columns))
    return list(common_columns)


@st.cache()
def build_inconsistencies_dataframe(df_source):
    df_inconsistent = (
        pd.concat(
            [
                df_source[~df_source.sporulation.isin([0, 1])].assign(
                    because="sporulation_oob"
                ),
                df_source[
                    ~(
                        (
                            (df_source.sporulation == 0)
                            & df_source.densite_sporulation.isna()
                        )
                        | (
                            (df_source.sporulation == 1)
                            & ~df_source.densite_sporulation.isna()
                        )
                    )
                ].assign(because="sporulation_ds_inc"),
                df_source[
                    ~(
                        df_source.densite_sporulation.isin(odd_numbers)
                        | df_source.densite_sporulation.isna()
                    )
                ].assign(because="densite_sporulation_oob"),
                df_source[df_source.necrose.isin([0, 1])].assign(because="necrose_oob"),
                df_source[
                    ~(
                        ((df_source.necrose == 1) & ~df_source.surface_necrosee.isna())
                        | ((df_source.necrose == 0) & df_source.surface_necrosee.isna())
                    )
                ].assign(because="necrose_sn_inc"),
                df_source[
                    ~(
                        ((df_source.necrose == 1) & ~df_source.taille_necrose.isna())
                        | ((df_source.necrose == 0) & df_source.taille_necrose.isna())
                    )
                ].assign(because="necrose_tn_inc"),
                df_source[
                    ~(
                        df_source.taille_necrose.isin(odd_numbers)
                        | df_source.taille_necrose.isna()
                    )
                ].assign(because="taille_necrose_oob"),
                df_source[
                    ~(
                        df_source.surface_necrosee.isin(odd_numbers)
                        | df_source.surface_necrosee.isna()
                    )
                ].assign(because="surface_necrosee_oob"),
                df_source[~df_source.oiv.isin(odd_numbers)].assign(because="oiv_oob"),
                df_source[
                    ~(
                        ((df_source.oiv == 9) & df_source.sporulation == 0)
                        | ((df_source.oiv != 9) & df_source.sporulation == 1)
                    )
                ].assign(because="oiv_s_inc"),
                df_source[~df_source.ligne.notna()].assign(because="ligne_oob"),
            ]
        )[["experiment", "sheet", "because"]]
        .sort_values(["experiment", "sheet", "because"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df_inconsistent = (
        df_inconsistent.assign(
            sporulation_oob=np.where(df_inconsistent.because == "sporulation_oob", 1, 0),
            sporulation_ds_inc=np.where(
                df_inconsistent.because == "sporulation_ds_inc", 1, 0
            ),
            densite_sporulation_oob=np.where(
                df_inconsistent.because == "densite_sporulation_oob", 1, 0
            ),
            necrose_oob=np.where(df_inconsistent.because == "necrose_oob", 1, 0),
            necrose_sn_inc=np.where(df_inconsistent.because == "necrose_sn_inc", 1, 0),
            necrose_tn_inc=np.where(df_inconsistent.because == "necrose_tn_inc", 1, 0),
            taille_necrose_oob=np.where(
                df_inconsistent.because == "taille_necrose_oob", 1, 0
            ),
            surface_necrosee_oob=np.where(
                df_inconsistent.because == "surface_necrosee_oob", 1, 0
            ),
            oiv_oob=np.where(df_inconsistent.because == "oiv_oob", 1, 0),
            oiv_s_inc=np.where(df_inconsistent.because == "oiv_s_inc", 1, 0),
            ligne_oob=np.where(df_inconsistent.because == "ligne_oob", 1, 0),
        )
        .drop(["because"], axis=1)
        .groupby(["experiment", "sheet"])
        .agg("sum")
        .reset_index(drop=False)
        .drop_duplicates()
    )

    df_inconsistent.to_csv(
        os.path.join(datain_path, "inconsistent_excels.csv"),
        index=False,
        sep=";",
    )

    return df_inconsistent


@st.cache()
def clean_merged_dataframe(df_source):
    return (
        df_source[
            (
                # sporulation must be 1 ou 0
                df_source.sporulation.isin([0, 1])
                # if sporulation is 0 , densite_sporulation must be NaN else it must be an odd number
                & (
                    ((df_source.sporulation == 0) & df_source.densite_sporulation.isna())
                    | (
                        (df_source.sporulation == 1)
                        & ~df_source.densite_sporulation.isna()
                    )
                )
                # densite_sporulation a number and not 0
                & (
                    df_source.densite_sporulation.isin(odd_numbers)
                    | df_source.densite_sporulation.isna()
                )
                # necrosis must be 1 ou 0
                & df_source.necrose.isin([0, 1])
                # if necrosis is 1 surface_necrosee & taille_necrose must not be none else they must
                & (
                    ((df_source.necrose == 1) & ~df_source.surface_necrosee.isna())
                    | ((df_source.necrose == 0) & df_source.surface_necrosee.isna())
                )
                & (
                    ((df_source.necrose == 1) & ~df_source.taille_necrose.isna())
                    | ((df_source.necrose == 0) & df_source.taille_necrose.isna())
                )
                # surface_necrosee & taille_necrose must be NaN or odd
                & (
                    df_source.taille_necrose.isin(odd_numbers)
                    | df_source.taille_necrose.isna()
                )
                & (
                    df_source.surface_necrosee.isin(odd_numbers)
                    | df_source.surface_necrosee.isna()
                )
                # OIV must be an odd number
                & df_source.oiv.isin(odd_numbers)
                # if OIV is 9 there must be no sporulation else there must be
                & (
                    ((df_source.oiv == 9) & df_source.sporulation == 0)
                    | ((df_source.oiv != 9) & df_source.sporulation == 1)
                )
                # line must not be NA
                & df_source.ligne.notna()
            )
        ]
        .assign(
            colonne=lambda x: x.colonne.astype("Int64"),
            necrose=lambda x: x.necrose.astype("Int64"),
            oiv=lambda x: x.oiv.astype("Int64"),
            sporulation=lambda x: x.sporulation.astype("Int64"),
            surface_necrosee=lambda x: x.surface_necrosee.astype("Int64"),
            densite_sporulation=lambda x: x.densite_sporulation.astype("Int64"),
            taille_necrose=lambda x: x.taille_necrose.astype("Int64"),
        )
        .drop_duplicates()
    )


@st.cache
def get_distant_excels():
    if os.path.isfile(excel_file_list_path):
        with open(excel_file_list_path, "r", encoding="UTF8") as f:
            files = f.read().split("?")
    else:
        files = [
            os.path.join(root, name)
            for root, _, files in tqdm(os.walk("Z:", topdown=False))
            for name in files
            if "_saisie" in name
            and "DM" in name
            and (name.endswith("xlsx") or name.endswith("xls"))
        ]
        pd.DataFrame(
            list(zip([os.path.basename(fn) for fn in files], files)),
            columns=["file", "path"],
        ).to_csv(os.path.join(datain_path, "imported_excels.csv"), sep=";")
        with open(excel_file_list_path, "w+", encoding="UTF8") as f:
            f.write("?".join(files))
    return files


@st.cache(suppress_st_warning=True)
def copy_excel_files(files):
    for i, file in enumerate(files):
        file_name = os.path.basename(file)
        pg_copy_excel_files.progress((i + 1) / len(files))
        if not file_name.startswith("~$") and not os.path.isfile(
            os.path.join(
                excel_file_path,
                file_name,
            )
        ):
            shutil.copy(src=file, dst=excel_file_path)


@st.cache
def filter_csvs():
    if os.path.isfile(path_to_df_result):
        return pd.read_csv(path_to_df_result)
    else:
        df_result = pd.DataFrame(
            columns=[
                "file",
                "sheet",
                "outcome",
                "comment",
                "csv_file_name",
            ]
        )

        def add_result(
            df,
            file,
            sheet,
            outcome,
            comment="success",
            csv_file_name=np.nan,
        ):
            return df.append(
                {
                    "file": file,
                    "sheet": sheet,
                    "outcome": outcome,
                    "comment": comment,
                    "csv_file_name": csv_file_name,
                },
                ignore_index=True,
            )

        def lower_dataframe(df):
            try:
                df.columns = df.columns.str.lower().str.replace(" ", "")
                for c in df.columns:
                    if c != "nomphoto" and df[c].dtype == object:
                        df[c] = df[c].str.lower().str.replace(" ", "")
            except:
                return False
            else:
                return df

        lcl_excel_files = [
            os.path.join(root, name)
            for root, _, files in os.walk(
                excel_file_path,
                topdown=False,
            )
            for name in files
            if name.endswith("_saisie.xlsx")
        ]

        for i, lcl_excel_file in enumerate(lcl_excel_files):
            pg_build_csv.progress((i + 1) / len(lcl_excel_files))
            tst_excel_file = pd.ExcelFile(lcl_excel_file)
            for sheet_name in tst_excel_file.sheet_names:
                df = lower_dataframe(df=tst_excel_file.parse(sheet_name=sheet_name))
                if df is False:
                    df_result = add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment="Corrupted dataframe",
                    )
                    continue
                header_loc = (
                    df[df == "numinc"].dropna(axis=1, how="all").dropna(how="all")
                )
                if header_loc.shape == (0, 0):
                    header_loc = (
                        df[df == "num"].dropna(axis=1, how="all").dropna(how="all")
                    )
                    if header_loc.shape == (0, 0):
                        df_result = add_result(
                            df=df_result,
                            file=os.path.basename(lcl_excel_file),
                            sheet=sheet_name,
                            outcome=False,
                            comment="No header",
                        )
                        continue
                df = lower_dataframe(
                    df=tst_excel_file.parse(
                        sheet_name,
                        skiprows=header_loc.index.item() + 1,
                        na_values=["", "NA", "na"],
                    )
                )
                if df is False:
                    df_result = add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment="Corrupted dataframe",
                    )
                    continue
                if (
                    res := check_list_in_list(
                        required_columns=needed_columns,
                        available_columns=df.columns.to_list(),
                    )
                ) is True:
                    csv_file_name = f"{Path(lcl_excel_file).stem}_{sheet_name}.csv"
                    df = df.assign(
                        exp=Path(lcl_excel_file).stem,
                        sheet=sheet_name,
                    ).dropna(subset=["nomphoto", "oiv"])[
                        needed_columns + ["exp", "sheet"]
                    ]
                    if df.shape[0] > 0:
                        df.to_csv(
                            os.path.join(oidium_extracted_csvs_path, csv_file_name),
                            index=False,
                        )
                        df_result = add_result(
                            df=df_result,
                            file=os.path.basename(lcl_excel_file),
                            sheet=sheet_name,
                            outcome=True,
                            csv_file_name=csv_file_name,
                        )
                    else:
                        df_result = add_result(
                            df=df_result,
                            file=os.path.basename(lcl_excel_file),
                            sheet=sheet_name,
                            outcome=False,
                            comment="Corrupted dataframe, failed to retrieve photos",
                        )
                else:
                    df_result = add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment=f"Missing columns: {res}",
                    )

        df_result.to_csv(path_to_df_result, index=False)
        return df_result


@st.cache()
def get_local_csvs():
    return [
        os.path.join(root, name)
        for root, _, files in os.walk(
            oidium_extracted_csvs_path,
            topdown=False,
        )
        for name in files
        if name.endswith(".csv")
    ]


@st.cache()
def build_raw_merged(lcl_csv_files):
    return pd.concat(
        [
            pd.read_csv(filepath)[get_common_columns(lcl_csv_files)]
            for filepath in lcl_csv_files
        ]
    ).rename(
        columns={
            "exp": "experiment",
            "sheet": "sheet",
            "oiv": "oiv",
            "nomphoto": "image_name",
            "s": "sporulation",
            "fn": "surface_necrosee",
            "n": "necrose",
            "sq": "densite_sporulation",
            "tn": "taille_necrose",
        }
    )


def build_dup_df(df):

    df_inverted_dup_check = df.drop(["colonne"], axis=1, errors="ignore").select_dtypes(
        exclude=object
    )

    df_dict = {
        k: df_inverted_dup_check[df_inverted_dup_check.oiv == k].drop(["oiv"], axis=1)
        for k in [1, 3, 5, 7, 9]
    }

    dup_df_lst = []
    pairs = []
    qtty = []
    dup_col_count = 0
    for i, j in [
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 9),
        (3, 5),
        (3, 7),
        (3, 9),
        (5, 7),
        (5, 9),
        (7, 9),
    ]:
        tmp_df = pd.merge(df_dict[i], df_dict[j], how="inner").drop_duplicates()
        tmp_df[f"{i}_{j}"] = True
        pairs.append(f"{i}_{j}")
        qtty.append(tmp_df.shape[0])
        if tmp_df.shape[0] > 0:
            dup_df_lst.append(tmp_df)
            dup_col_count += 1
    if len(dup_df_lst) > 0:
        df_dup = (
            reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=df_inverted_dup_check.drop(["oiv"], axis=1).columns.to_list(),
                    how="outer",
                ),
                dup_df_lst,
            )
            .drop_duplicates()
            .sort_values(df_inverted_dup_check.drop(["oiv"], axis=1).columns.to_list())
            .reset_index(drop=True)
        )
        df_dup = (
            df_dup[df_dup.isnull().sum(axis=1) < dup_col_count - 1]
            .dropna(axis=1, how="all")
            .reset_index(drop=True)
        )
    else:
        df_dup = pd.DataFrame()

    return {"df_dup": df_dup, "pairs": pairs, "count": qtty}


@st.cache()
def build_sbs_plsda(df_src, df_dup):
    df_sheet_plsda = pd.DataFrame(columns=["experiment", "sheet", "row_count", "score"])

    df_dup_compare = df_dup.select_dtypes(exclude=object)
    for _, row in df_src[["experiment", "sheet"]].drop_duplicates().iterrows():
        try:
            df = (
                df_src[
                    (df_src.experiment == row["experiment"])
                    & (df_src.sheet == row["sheet"])
                ]
                .select_dtypes(exclude=object)
                .drop(["colonne"], axis=1)
                .drop_duplicates()
            )
            tmp_df = pd.merge(df, df_dup_compare, how="inner").drop_duplicates()
            X = df.drop(["oiv"], axis=1)
            y = df.oiv
            X = StandardScaler().fit(X).transform(X)
            cur_pls_da = PLSRegression(n_components=X.shape[1])
            cur_pls_da.fit(X, y).transform(X)

            df_sheet_plsda = df_sheet_plsda.append(
                {
                    "experiment": row["experiment"],
                    "sheet": row["sheet"],
                    "row_count": df.shape[0],
                    "score": cur_pls_da.score(X, df.oiv),
                    "dup_count": tmp_df.shape[0],
                    "dup_rate": tmp_df.shape[0] / df.shape[0],
                },
                ignore_index=True,
            )
        except:
            df_sheet_plsda = df_sheet_plsda.append(
                {"experiment": row["experiment"], "sheet": row["sheet"]},
                ignore_index=True,
            )
    return df_sheet_plsda


@st.cache()
def cache_build_dup_df(df):
    return build_dup_df(df)


st.markdown("# Leaf Disk Collate Ground Truth")

col_target, col_explain = st.columns(2)

with col_target:
    st.markdown(
        """
    This notebook will:
    - Retrieve all available Excel files
    - Translate them to CSV and merge them
    - Build models to asses the possibility of predicting OIV from various visual variables
    """
    )
    st.markdown(
        """
    We need:
    - Base python libraries for file management
    - tqdm for progress tracking
    - Pandas and Numpy for the dataframes
    - SkLearn for statistics
    - Plotly for ... plotting    
    """
    )

with col_explain:

    st.markdown(
        """
    Functions needed to:
    - Check that the dataframe has at least the needed columns
    - Plot model variance
    - Plot an histogram of the variables needed for the OIV so inconsistencies can be detected
    - Generate categorical OIV from dataframe
    """
    )

    st.markdown(
        f"""
    Constants:
    - Path to datain: {os.path.abspath(datain_path)}
    - Path to distant Excel files: {os.path.abspath(excel_file_path)}
    - Path to local EXcel files: {os.path.abspath(oidium_extracted_csvs_path)}
    - Path to extracted CSVs: {os.path.abspath(excel_file_list_path)}
    - Path to individual CSV generation result: {os.path.abspath(path_to_df_result)}
    - Needed columns: {needed_columns}
    """
    )

st.markdown("## What is OIV and how do we want to predict it")

col_desc_oiv, col_desc_variables = st.columns(2)

with col_desc_oiv:
    st.markdown("### OIV")

    st.markdown(
        "OIV 452-2 is a standard to evaluate resistance to powdery mildew in vine disk leafs"
    )
    st.markdown(
        """
        > &mdash; From OIV the 452-2 specification.
        >
        >  Characteristic: Leaf: degree of resistance to Plasmopara (leaf disc test)  
        >  Notes:
        >  1: very little 3:little 5:medium 7:high 9:very high   
        >  Observation during the whole vegetation period, as long as there are young leaves, on vines not treated with
        >  chemicals.
        >  Because the zoospores penetrate through the stomata, the leaf discs have to be placed with the lower surface up.
        >  Using a standardized spore suspension with 25000 spores/ml (counting chamber), a pipette is used to place 40µl
        >  or 1000 spores on each leaf disc.
        >  Incubation: in complete darkness (aluminum coat), room temperature, 4 days.
        >  Remark: if the inoculum remains on the leaf disc too long, lesions are produced. Therefore, 24 hours after
        >  inoculation, the spore suspension has to be removed by blotting with a filter paper. 
        """
    )
    st.image(os.path.join(datain_path, "images", "OIV_examples.png"))
    st.warning(
        "OIV 452-2 is a resistance scale, higher note means less disease phenotype"
    )

with col_desc_variables:
    st.markdown("### Other variables")
    st.markdown("Other variables with which we want to predict OIV 452-2")
    st.image(os.path.join(datain_path, "images", "oiv_452-1_desc.png"))
    st.markdown(oiv_452_spec_req)


st.markdown("### The aim of this dashboard")
st.markdown(
    "Can we predict the OIV from the other variables, on other words is there a link between the variables ond the OIV?"
)

st.markdown("## Build dataframe")


lcl_csv_files = get_local_csvs()

st.markdown("### Retrieve distant Excels")
st.markdown(
    """
Get all related file's path in the distant server.  
Experiements are stored by year and by experiment, the excels files with data are Excel classifiers which contain "saisie", 
we're going to parse all the folders year by year and retrieve the files.

- Files containing DM for domny mildew, ie mildiou, are selected for OIV analysis
- Files containing PM for powdery mildew, ie oïdium, are discarded
"""
)

if os.path.isfile(path_to_df_result) is False:
    files = get_distant_excels()

    st.write(files)

    st.write("Copying files")
    pg_copy_excel_files = st.progress(0)

    copy_excel_files(files)

    pg_copy_excel_files.progress(1.0)
else:
    st.write("Excel files already parsed")
st.success("")

st.markdown("### Build CSVs")

st.markdown(
    """
We look for 2 particular headers, sheets will be discarded if:
- the header is not found
- the dataframe is corrupted, ie unable to find images or a column is malformed
"""
)

st.write("Building CSVs")

pg_build_csv = st.progress(0)

df_result = filter_csvs()

pg_build_csv.progress(1.0)

lcl_csv_files = [
    os.path.join(oidium_extracted_csvs_path, filename)
    for filename in df_result.csv_file_name.dropna().to_list()
]

st.write("Extracted CSVs")
st.write(lcl_csv_files)

st.write(f"A sample file: {lcl_csv_files[13]}")
sample_csv = st.selectbox(
    label="Select a CSV to view:",
    options=["Select one"] + lcl_csv_files,
    index=0,
    format_func=lambda x: os.path.basename(x),
)
if sample_csv != "Select one":
    st.dataframe(pd.read_csv(sample_csv))


st.markdown(
    """
**Some CSVs where rejected:**
- Some are experiment description with no data
- Some have no images
- Some are corrupted, ie, it was impossible to read them
- ...
"""
)

fig = px.histogram(
    data_frame=df_result.sort_values(["comment"]),
    x="comment",
    color="comment",
    width=1400,
    height=400,
    text_auto=True,
).update_layout(
    font=dict(
        family="Courier New, monospace",
        size=18,
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
        text=f"From {df_result.shape[0]} sheets available {df_result[df_result.comment == 'success'].shape[0]} were successfully loaded",
        textangle=0,
        xanchor="left",
        xref="paper",
        yref="paper",
    )
)

st.plotly_chart(fig)

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
    .drop(["csv_file_name"], axis=1)
    .reset_index(drop=True)
)

df_corrupted.to_csv(
    os.path.join(datain_path, "corrupted_excels.csv"),
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
st.markdown(oiv_452_spec_req)
st.plotly_chart(plot_inconsistencies(df_raw_merged, sort_values=False))
st.warning("Various rows are inconsistent")
st.markdown("After removing inconsistent lines we get a new consistent dataframe")
df_merged = clean_merged_dataframe(df_raw_merged)
st.dataframe(df_merged.head(50))
st.markdown(df_merged.shape)
st.plotly_chart(plot_inconsistencies(df_merged))
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

st.markdown(
    """
This has not been successful, were going o try switching from a resistance scale to a susceptibility scale, 
this allows us to keep all dimensions for all observations.  
If we invert the axes to have sensitivity scale instead of a resistance scale, 
this will allow us to include all previously removed NaN contained rows as all OIV 5 rows
"""
)

col_inv_df, con_inv_num_df = st.columns([3, 2])


df_inverted = (
    df_merged.assign(
        surface_necrosee=lambda x: 10 - x.surface_necrosee,
        densite_sporulation=lambda x: 10 - x.densite_sporulation,
        taille_necrose=lambda x: 10 - x.taille_necrose,
    )
    .assign(
        surface_necrosee=lambda x: x.surface_necrosee.fillna(0),
        densite_sporulation=lambda x: x.densite_sporulation.fillna(0),
        taille_necrose=lambda x: x.taille_necrose.fillna(0),
        sporulation=lambda x: x.sporulation.fillna(0),
    )
    .drop_duplicates()
    .sort_values(
        [
            "oiv",
            "experiment",
            "sheet",
        ]
    )
)

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
        px.imshow(
            df_inv_num[df_inv_num.oiv == 1].drop_duplicates().reset_index(drop=True),
            color_continuous_scale=px.colors.sequential.BuPu,
            height=col_oiv_height,
            width=col_oiv_width,
            title="OIV 1",
        ),
    )

with col_oiv_3:
    st.plotly_chart(
        px.imshow(
            df_inv_num[df_inv_num.oiv == 3].drop_duplicates().reset_index(drop=True),
            color_continuous_scale=px.colors.sequential.BuPu,
            height=col_oiv_height,
            width=col_oiv_width,
            title="OIV 3",
        ),
    )

with col_oiv_5:
    st.plotly_chart(
        px.imshow(
            df_inv_num[df_inv_num.oiv == 5].drop_duplicates().reset_index(drop=True),
            color_continuous_scale=px.colors.sequential.BuPu,
            height=col_oiv_height,
            width=col_oiv_width,
            title="OIV 5",
        ),
    )

col_oiv_7, col_oiv_9, col_oiv_avg = st.columns(3)

with col_oiv_7:
    st.plotly_chart(
        px.imshow(
            df_inv_num[df_inv_num.oiv == 7].drop_duplicates().reset_index(drop=True),
            color_continuous_scale=px.colors.sequential.BuPu,
            height=col_oiv_height,
            width=col_oiv_width,
            title="OIV 7",
        ),
    )

with col_oiv_9:
    st.plotly_chart(
        px.imshow(
            df_inv_num[df_inv_num.oiv == 9].drop_duplicates().reset_index(drop=True),
            color_continuous_scale=px.colors.sequential.BuPu,
            height=col_oiv_height,
            width=col_oiv_width,
            title="OIV 9",
        ),
    )
with col_oiv_avg:
    st.plotly_chart(
        px.imshow(
            df_inv_num.groupby(["oiv"]).mean().reset_index(drop=False),
            color_continuous_scale=px.colors.sequential.Viridis,
            height=col_oiv_height,
            width=col_oiv_width,
            title="Average values for all OIVs",
            text_auto=True,
        ),
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

    pca_data = PCA()
    x_inv_new = pca_data.fit_transform(Xi)

    fig = px.scatter(
        x=x_inv_new[:, inv_x_comp],
        y=x_inv_new[:, inv_y_comp],
        color=yi.astype(str),
        height=700,
        width=800,
        title="Inverted PCA 2D",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )

    fig.update_xaxes(
        range=[
            x_inv_new[:, inv_x_comp].min() - 0.5,
            x_inv_new[:, inv_x_comp].max() + 0.5,
        ]
    )
    fig.update_yaxes(
        range=[
            x_inv_new[:, inv_y_comp].min() - 0.5,
            x_inv_new[:, inv_y_comp].max() + 0.5,
        ]
    )
    st.plotly_chart(fig)

with inv_splsda:
    st.markdown("#### sPLSDA")
    pls_data_all_inv = PLSRegression(n_components=Xi.shape[1])
    x_new = pls_data_all_inv.fit(Xi, yi).transform(Xi)

    fig = px.scatter(
        x=pls_data_all_inv.x_scores_[:, inv_x_comp],
        y=pls_data_all_inv.x_scores_[:, inv_y_comp],
        color=yi.astype(str),
        height=700,
        width=800,
        title="Inverted sPLS-DA",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )
    fig.update_xaxes(
        range=[
            pls_data_all_inv.x_scores_[:, inv_x_comp].min() - 0.5,
            pls_data_all_inv.x_scores_[:, inv_x_comp].max() + 0.5,
        ]
    )
    fig.update_yaxes(
        range=[
            pls_data_all_inv.x_scores_[:, inv_y_comp].min() - 0.5,
            pls_data_all_inv.x_scores_[:, inv_y_comp].max() + 0.5,
        ]
    )
    st.plotly_chart(fig)
    st.markdown(f"**sPLSDA score**: {pls_data_all_inv.score(Xi, yi)}")

st.markdown("### Check overlapping")

st.write(
    "Some observations seem to overlap, we're going to check that one point in the vectorial space codes only one OIV"
)

d = cache_build_dup_df(df_inverted)

df_dup = d["df_dup"]
pairs = d["pairs"]
qtty = d["count"]

col_dup_df, col_dup_unique, col_dup_count = st.columns([4, 2, 1])

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

print(df_inverted.dtypes)

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
    print_dataframe_and_shape(build_dup_df(df_es)["df_dup"])

with sbs_sng_scatter:
    st.markdown("#### sPLS-DA")
    X = df_es.drop(["oiv"], axis=1)
    y = df_es.oiv
    X = StandardScaler().fit(X).transform(X)
    es_pls_da = PLSRegression(n_components=X.shape[1]).fit(X, y)
    fig = px.scatter(
        x=es_pls_da.x_scores_[:, 0],
        y=es_pls_da.x_scores_[:, 1],
        color=y.astype(str),
        height=700,
        width=700,
        title=f"sPLS-DA for experiment {exp} sheet {sheet}score: {es_pls_da.score(X, y)}",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"), opacity=0.7),
        selector=dict(mode="markers"),
    )
    fig.update_xaxes(
        range=[
            es_pls_da.x_scores_[:, 0].min() - 0.5,
            es_pls_da.x_scores_[:, 0].max() + 0.5,
        ]
    )
    fig.update_yaxes(
        range=[
            es_pls_da.x_scores_[:, 1].min() - 0.5,
            es_pls_da.x_scores_[:, 1].max() + 0.5,
        ]
    )
    st.plotly_chart(fig)
