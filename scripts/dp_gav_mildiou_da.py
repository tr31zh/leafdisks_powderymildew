import os
from typing import Text
import warnings
import itertools

import pandas as pd

import datapane as dp

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
import gav_mildiou_plotly as gop


pd.options.plotting.backend = "plotly"
pd.options.display.float_format = "{:4,.2f}".format

warnings.simplefilter("ignore")


def warning(text: str) -> str:
    return """
<html>
    <style type='text/css'>
        @keyframes example {
            0%   {color: #EEE;}
            25%  {color: #EC4899;}
            50%  {color: #8B5CF6;}
            100% {color: #EF4444;}
        }
        #container {
            background: #1F2937;
            padding: 10em;
        }
        h1 {
            color:#eee;
            animation-name: example;
            animation-duration: 4s;
            animation-iteration-count: infinite;
        }
    </style>
    <div id="container">
      <h1>text</h1>
    </div>
</html>
""".replace(
        "text", text
    )


lcl_csv_files = gof.get_local_csvs()


dp.Report(
    dp.Text(f"{goc.lvl_1_header} OIV: Existing annaotation data analysis"),
    dp.Group(
        dp.Text(got.txt_target),
        dp.Text(got.txt_libraries),
        columns=2,
    ),
    dp.Text(f"{goc.lvl_2_header} What is OIV and how do we want to predict it"),
    dp.Text(got.txt_oiv_452_spec),
    dp.Media(file=os.path.join(goc.datain_path, "images", "OIV_examples.png")),
    dp.HTML(
        warning(
            "OIV 452-2 is a resistance scale, higher note means less disease phenotype"
        )
    ),
    dp.Text(f"{goc.lvl_3_header} Other variables"),
    dp.Text("Other variables with which we want to predict OIV 452-2"),
    dp.Media(file=os.path.join(goc.datain_path, "images", "oiv_452-1_desc.png")),
    dp.Text(got.txt_oiv_452_spec_header),
    dp.Text(got.txt_what_we_want),
    dp.Text(f"{goc.lvl_2_header} Build dataframe"),
    dp.Text(f"{goc.lvl_3_header} Retrieve distant Excels"),
    dp.Text(got.txt_get_excels),
).save(path=os.path.join(".", "data_out", "reports", "mildiou-report.html"))
