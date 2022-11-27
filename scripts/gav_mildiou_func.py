from pathlib import Path
import os
import shutil
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_flavor as pf

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

import gav_mildiou_const as goc


def check_list_in_list(required_columns, available_columns):
    failures = []
    for rc in required_columns:
        if rc not in available_columns:
            failures.append(rc)

    return True if len(failures) == 0 else failures


def get_oiv_cat(df):
    return df.oiv.astype(str)


def get_common_columns(csv_files):
    common_columns = set(read_dataframe(csv_files[0]).columns.to_list())
    columns_occ = {}
    for filepath in csv_files:
        cu_columns = read_dataframe(filepath).columns.to_list()
        for c in cu_columns:
            if c in columns_occ:
                columns_occ[c] += 1
            else:
                columns_occ[c] = 1
        common_columns = common_columns.intersection(set(cu_columns))
    return list(common_columns)


consistency_checks_list = [
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


def ensure_folder(forced_path, return_string: bool = True):
    path = forced_path.parent
    if path.is_dir() is False:
        path.mkdir(parents=True, exist_ok=True)
    return str(forced_path) if return_string is True else forced_path


def read_dataframe(path) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=str(path), sep=";")


def write_dataframe(df: pd.DataFrame, path) -> pd.DataFrame:
    df.to_csv(path_or_buf=ensure_folder(path, return_string=True), sep=";", index=False)
    return df


def consistency_checks(df_src):
    return {
        "sporulation_oob": df_src.sporulation.isin([0, 1]),
        "sporulation_ds_inc": (
            ((df_src.sporulation == 0) & df_src.densite_sporulation.isna())
            | ((df_src.sporulation == 1) & ~df_src.densite_sporulation.isna())
        ),
        "densite_sporulation_oob": (
            df_src.densite_sporulation.isin(goc.odd_numbers)
            | df_src.densite_sporulation.isna()
        ),
        "necrose_oob": df_src.necrose.isin([0, 1]),
        "necrose_sn_inc": (
            ((df_src.necrose == 1) & ~df_src.surface_necrosee.isna())
            | ((df_src.necrose == 0) & df_src.surface_necrosee.isna())
        ),
        "necrose_tn_inc": (
            ((df_src.necrose == 1) & ~df_src.taille_necrose.isna())
            | ((df_src.necrose == 0) & df_src.taille_necrose.isna())
        ),
        "taille_necrose_oob": (
            df_src.taille_necrose.isin(goc.odd_numbers) | df_src.taille_necrose.isna()
        ),
        "surface_necrosee_oob": (
            df_src.surface_necrosee.isin(goc.odd_numbers)
            | df_src.surface_necrosee.isna()
        ),
        "oiv_oob": df_src.oiv.isin(goc.odd_numbers),
        "oiv_s_inc": (
            ((df_src.oiv == 9) & df_src.sporulation == 0)
            | ((df_src.oiv != 9) & df_src.sporulation == 1)
        ),
        "ligne_oob": df_src.ligne.notna(),
    }


def build_inconsistencies_dataframe(df_source):
    if goc.inconsistent_sheets.is_file():
        return read_dataframe(goc.inconsistent_sheets)
    else:
        checks = consistency_checks(df_src=df_source)
        df_inconsistent = (
            pd.concat([df_source[~v].assign(because=k) for k, v in checks.items()])[
                ["experiment", "sheet", "because"]
            ]
            .sort_values(["experiment", "sheet", "because"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_inconsistent = (
            df_inconsistent.assign(
                sporulation_oob=np.where(
                    df_inconsistent.because == "sporulation_oob", 1, 0
                ),
                sporulation_ds_inc=np.where(
                    df_inconsistent.because == "sporulation_ds_inc", 1, 0
                ),
                densite_sporulation_oob=np.where(
                    df_inconsistent.because == "densite_sporulation_oob", 1, 0
                ),
                necrose_oob=np.where(df_inconsistent.because == "necrose_oob", 1, 0),
                necrose_sn_inc=np.where(
                    df_inconsistent.because == "necrose_sn_inc", 1, 0
                ),
                necrose_tn_inc=np.where(
                    df_inconsistent.because == "necrose_tn_inc", 1, 0
                ),
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

        return write_dataframe(df=df_inconsistent, path=goc.inconsistent_sheets)


def clean_merged_dataframe(df_source):
    if goc.clean_merged.is_file():
        return read_dataframe(goc.clean_merged)
    else:
        checks = consistency_checks(df_src=df_source)
        df_clean_merged = (
            df_source[
                (
                    checks["sporulation_oob"]
                    & checks["sporulation_ds_inc"]
                    & checks["densite_sporulation_oob"]
                    & checks["necrose_oob"]
                    & checks["necrose_sn_inc"]
                    & checks["necrose_tn_inc"]
                    & checks["taille_necrose_oob"]
                    & checks["surface_necrosee_oob"]
                    & checks["oiv_oob"]
                    & checks["oiv_s_inc"]
                    & checks["ligne_oob"]
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
        return write_dataframe(df=df_clean_merged, path=goc.clean_merged)


def get_distant_excels():
    if goc.distant_excels.is_file():
        return read_dataframe(goc.distant_excels)
    else:
        if goc.distant_excel_file_path.is_dir() is False:
            raise FileNotFoundError("Unable to acces distant server")
        files = [
            os.path.join(root, name)
            for root, _, files in tqdm(
                os.walk(
                    str(goc.distant_excel_file_path),
                    topdown=False,
                    followlinks=True,
                ),
                desc="Looking for distant Excels",
            )
            for name in files
            if "_saisie" in name
            and "DM" in name
            and (name.endswith("xlsx") or name.endswith("xls"))
        ]
        return write_dataframe(
            pd.DataFrame(
                list(zip([os.path.basename(fn) for fn in files], files)),
                columns=["file", "path"],
            ),
            goc.distant_excels,
        )


def copy_excel_files(files):
    if goc.excel_file_path.is_dir() is False:
        goc.excel_file_path.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc="Copying Excel files"):
        file_name = os.path.basename(file)
        ensure_folder(goc.excel_file_path)
        dst_file = goc.excel_file_path.joinpath(file_name)
        if file_name.startswith("~$") is False and dst_file.is_file() is False:
            shutil.copy(src=file, dst=str(dst_file))


def _add_result(
    df,
    file,
    sheet,
    outcome,
    comment="success",
    csv_file_name=np.nan,
):
    return pd.concat(
        [
            df,
            pd.DataFrame(
                data={
                    "file": [file],
                    "sheet": [sheet],
                    "outcome": [outcome],
                    "comment": [comment],
                    "csv_file_name": [csv_file_name],
                }
            ),
        ],
        # axis=0,
        ignore_index=True,
    )


def _lower_dataframe(df):
    try:
        df.columns = df.columns.str.lower().str.replace(" ", "")
        for c in df.columns:
            if c != "photo" and df[c].dtype == object:
                df[c] = df[c].str.lower().str.replace(" ", "")
    except:
        return False
    else:
        return df


def filter_csvs():
    if os.path.isfile(goc.csv_filter_result):
        return read_dataframe(goc.csv_filter_result)
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

        lcl_excel_files = [
            os.path.join(root, name)
            for root, _, files in os.walk(
                goc.excel_file_path,
                topdown=False,
            )
            for name in files
            if "_saisie" in name
            and "DM" in name
            and (name.endswith("xlsx") or name.endswith("xls"))
        ]

        for lcl_excel_file in tqdm(lcl_excel_files, desc="Filtering Escel sheets"):
            tst_excel_file = pd.ExcelFile(lcl_excel_file)
            for sheet_name in tst_excel_file.sheet_names:
                df = _lower_dataframe(df=tst_excel_file.parse(sheet_name=sheet_name))
                if df is False:
                    df_result = _add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment="Corrupted dataframe",
                    )
                    continue

                for tag in ["numinc", "num", "rep"]:
                    if tag in df.columns.to_list():
                        df = _lower_dataframe(df.iloc[:, df.columns.get_loc(tag) :])
                        break
                    header_loc = (
                        lambda x, y: (
                            x[x == y].dropna(axis=1, how="all").dropna(how="all")
                        )
                    )(df, tag)
                    if header_loc.shape != (0, 0):
                        df = _lower_dataframe(
                            df=tst_excel_file.parse(
                                sheet_name,
                                skiprows=header_loc.index.item() + 1,
                                na_values=["", "NA", "na"],
                            )
                        )
                        break
                else:
                    df_result = _add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment="No header",
                    )
                    continue
                if df is False:
                    df_result = _add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment="Corrupted dataframe",
                    )
                    continue
                if (
                    res := check_list_in_list(
                        required_columns=goc.needed_columns,
                        available_columns=df.columns.to_list(),
                    )
                ) is True:
                    csv_file_name = f"{Path(lcl_excel_file).stem}_{sheet_name}.csv"
                    df = df.assign(
                        exp=Path(lcl_excel_file).stem,
                        sheet=sheet_name,
                    ).dropna(subset=["photo", "oiv"])[
                        goc.needed_columns + ["exp", "sheet"]
                    ]
                    if df.shape[0] > 0:
                        write_dataframe(
                            df=df.rename(
                                columns={
                                    "exp": "experiment",
                                    "sheet": "sheet",
                                    "oiv": "oiv",
                                    "photo": "image_name",
                                    "s": "sporulation",
                                    "fn": "surface_necrosee",
                                    "n": "necrose",
                                    "sq": "densite_sporulation",
                                    "tn": "taille_necrose",
                                }
                            ),
                            path=goc.mildiou_extracted_csvs_path.joinpath(
                                csv_file_name
                            ),
                        )
                        df_result = _add_result(
                            df=df_result,
                            file=os.path.basename(lcl_excel_file),
                            sheet=sheet_name,
                            outcome=True,
                            csv_file_name=csv_file_name,
                        )
                    else:
                        df_result = _add_result(
                            df=df_result,
                            file=os.path.basename(lcl_excel_file),
                            sheet=sheet_name,
                            outcome=False,
                            comment="Corrupted dataframe, failed to retrieve photos",
                        )
                else:
                    df_result = _add_result(
                        df=df_result,
                        file=os.path.basename(lcl_excel_file),
                        sheet=sheet_name,
                        outcome=False,
                        comment=f"Missing columns: {res}",
                    )

        return write_dataframe(df_result.sort_values(["file"]), goc.csv_filter_result)


def get_local_csvs():
    return [
        os.path.join(root, name)
        for root, _, files in os.walk(
            goc.mildiou_extracted_csvs_path,
            topdown=False,
        )
        for name in files
        if name.endswith(".csv")
    ]


def build_raw_merged(lcl_csv_files):
    if os.path.isfile(goc.raw_merged):
        return read_dataframe(goc.raw_merged)
    else:
        return write_dataframe(
            df=pd.concat(
                [
                    read_dataframe(filepath)[get_common_columns(lcl_csv_files)]
                    for filepath in lcl_csv_files
                ]
            ),
            path=goc.raw_merged,
        )


def build_dup_df(df):

    df_inverted_dup_check = df.drop(
        ["colonne"],
        axis=1,
        errors="ignore",
    ).select_dtypes(exclude=object)

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
        tmp_df[(i, j)] = True
        pairs.append((i, j))
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

            df_sheet_plsda = pd.concat(
                [
                    df_sheet_plsda,
                    pd.DataFrame(
                        data={
                            "experiment": [row["experiment"]],
                            "sheet": [row["sheet"]],
                            "row_count": [df.shape[0]],
                            "score": [cur_pls_da.score(X, df.oiv)],
                            "dup_count": [tmp_df.shape[0]],
                            "dup_rate": [tmp_df.shape[0] / df.shape[0]],
                        }
                    ),
                ],
                # axis=0,
                ignore_index=True,
            )
        except:
            df_sheet_plsda = pd.concat(
                [
                    df_sheet_plsda,
                    pd.DataFrame(
                        data={
                            "experiment": [row["experiment"]],
                            "sheet": [row["sheet"]],
                        }
                    ),
                ],
                ignore_index=True,
            )
    return df_sheet_plsda


def build_sbs_dup_df(df_src):
    return pd.concat(
        [
            build_dup_df(
                df_src[
                    (
                        (df_src.experiment == row["experiment"])
                        & (df_src.sheet == row["sheet"])
                    )
                ]
                .select_dtypes(exclude=object)
                .drop(["colonne"], axis=1)
                .drop_duplicates()
            )["df_dup"].assign(experiment=row["experiment"], sheet=row["sheet"])
            for _, row in df_src[["experiment", "sheet"]].drop_duplicates().iterrows()
        ]
    )


def invert_axis(df_src, fill_val: int = 1):
    return (
        df_src.assign(
            surface_necrosee=lambda x: 10 - x.surface_necrosee,
            densite_sporulation=lambda x: 10 - x.densite_sporulation,
            taille_necrose=lambda x: 10 - x.taille_necrose,
        )
        .assign(
            surface_necrosee=lambda x: x.surface_necrosee.fillna(fill_val),
            densite_sporulation=lambda x: x.densite_sporulation.fillna(fill_val),
            taille_necrose=lambda x: x.taille_necrose.fillna(fill_val),
            sporulation=lambda x: x.sporulation.fillna(fill_val),
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


def sheet_filtering_out_df(df):
    return (
        df[
            df.comment.isin(
                [
                    "Corrupted dataframe",
                    "Corrupted dataframe, failed to retrieve photos",
                ]
            )
        ]
        .drop(["csv_file_name", "outcome"], axis=1)
        .reset_index(drop=True)
    )


def build_all_dataframes() -> dict:
    result = {}

    result["files"] = get_distant_excels()
    copy_excel_files(result["files"].path.to_list())

    result["result"] = filter_csvs()
    lcl_csv_files = [
        os.path.join(goc.mildiou_extracted_csvs_path, filename)
        for filename in result["result"].csv_file_name.dropna().to_list()
    ]

    result["raw_merged"] = build_raw_merged(lcl_csv_files)
    result["merged"] = clean_merged_dataframe(result["raw_merged"])

    result["num"] = (
        result["merged"]
        .drop(["colonne"], axis=1)
        .dropna()
        .select_dtypes(exclude=object)
        .drop_duplicates()
    )
    num_cols = result["num"].columns
    num_cols = [
        "sporulation",
        "densite_sporulation",
        "necrose",
        "surface_necrosee",
        "taille_necrose",
        "oiv",
    ]
    result["num"] = result["num"][num_cols].sort_values(
        ["oiv", "sporulation", "necrose"]
    )

    result["inverted"] = invert_axis(result["merged"])
    result["inverted"] = result["inverted"][
        [result["inverted"].columns[i] for i in [9, 8, 4, 6, 0, 3, 2, 10, 5, 7, 1]]
    ].sort_values(["oiv", "sporulation", "necrose"])

    result["inv_num"] = (
        result["inverted"]
        .drop(["colonne"], axis=1)
        .select_dtypes(exclude=object)
        .drop_duplicates()
        .sort_values(
            [
                "oiv",
                "necrose",
                "taille_necrose",
                "surface_necrosee",
                "sporulation",
                "densite_sporulation",
            ]
        )
    )

    return result
