from pathlib import Path

root_folder = Path(__file__).parent.parent

datain_path = root_folder.joinpath("data_in")
dataout_path = root_folder.joinpath("data_out")
excel_file_path = datain_path.joinpath("mildiou_source_excels")
distant_excel_file_path = datain_path.joinpath("gav_phenotypage")
mildiou_extracted_csvs_path = datain_path.joinpath("mildiou_extracted_csvs")
distant_excels_df = datain_path.joinpath("imported_excels.csv")
path_to_df_result = datain_path.joinpath("extracted_csv_files.csv")

odd_numbers = [1, 3, 5, 7, 9]
needed_columns = ["nomphoto", "oiv", "s", "sq", "n", "fn", "tn", "ligne", "colonne"]

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
