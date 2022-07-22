import os

import gav_oidium_const as gof

txt_oiv_452_spec_req = """
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

txt_oiv_452_spec = """
OIV 452-2 is a standard to evaluate resistance to powdery mildew in vine disk leafs

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

txt_target = """
**This notebook will**:
- Retrieve all available Excel files
- Translate them to CSV and merge them
- Build models to asses the possibility of predicting OIV from various visual variables

**We need**:
- Base python libraries for file management
- Pandas and Numpy for the dataframes
- SkLearn for statistics
- Plotly for ... plotting    
"""

txt_python_need = f"""
**Functions needed to**:
- Check that the dataframe has at least the needed columns
- Plot model variance
- Plot an histogram of the variables needed for the OIV so inconsistencies can be detected
- Generate categorical OIV from dataframe
**Constants**:
- Path to datain: {os.path.abspath(gof.datain_path)}
- Path to distant Excel files: {os.path.abspath(gof.excel_file_path)}
- Path to local EXcel files: {os.path.abspath(gof.oidium_extracted_csvs_path)}
- Path to extracted CSVs: {os.path.abspath(gof.excel_file_list_path)}
- Path to individual CSV generation result: {os.path.abspath(gof.path_to_df_result)}
- Needed columns: {gof.needed_columns}
"""
