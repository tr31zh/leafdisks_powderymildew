from pathlib import Path

# Folders
root_folder = Path(__file__).parent.parent
datain_path = root_folder.joinpath("data_in")
dataout_path = root_folder.joinpath("data_out")
excel_file_path = datain_path.joinpath("mildiou_source_excels")
distant_excel_file_path = datain_path.joinpath("gav_phenotypage")
dataframes_path = datain_path.joinpath("dataframes")

# Dataframes
distant_excels = dataframes_path.joinpath("distant_excels_df.csv")
clean_merged = dataframes_path.joinpath("clean_merged.csv")
csv_filter_result = dataframes_path.joinpath("csv_filter_result.csv")
inconsistent_sheets = dataframes_path.joinpath("inconsistent_sheets.csv")
clean_merged = dataframes_path.joinpath("clean_merged.csv")
raw_merged = dataframes_path.joinpath("raw_merged.csv")

mildiou_extracted_csvs_path = datain_path.joinpath("mildiou_extracted_csvs")

odd_numbers = [1, 3, 5, 7, 9]
needed_columns = ["photo", "oiv", "s", "sq", "n", "fn", "tn", "ligne", "colonne"]

three_plot_width = 600
three_plot_height = 500
four_plot_width = 500
four_plot_height = 400
two_plot_width = 800
two_plot_height = 700
large_plot_width = 1400
large_plot_height = 1000

lvl_1_header = "#"
lvl_2_header = "##"
lvl_3_header = "###"
lvl_4_header = "####"
lvl_5_header = "#####"
lvl_6_header = "######"
lvl_7_header = "#######"
