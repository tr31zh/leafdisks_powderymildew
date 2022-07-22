import os

datain_path = os.path.join(".", "data_in")
excel_file_path = os.path.join(datain_path, "oidium_source_excels", "")
oidium_extracted_csvs_path = os.path.join(datain_path, "oidium_extracted_csvs", "")
excel_file_list_path = os.path.join(excel_file_path, "excel_list.txt")
path_to_df_result = os.path.join(datain_path, "extracted_csv_files.csv")
odd_numbers = [1, 3, 5, 7, 9]
needed_columns = ["nomphoto", "oiv", "s", "sq", "n", "fn", "tn", "ligne", "colonne"]
