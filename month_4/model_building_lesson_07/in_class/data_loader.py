# import pandas as pd
# import os
# class DataLoader:
#     """
#     Class to load and merge CSV datasets from multiple league folders inside Data/Raw_Data.
#     """
    
#     def __init__(self, raw_data_path="Data/Raw_Data/"):
#         self.raw_data_path = raw_data_path

#     def load_league_files(self):
#         """
#         Loads all CSV files from each league folder and returns a list of DataFrames.
#         """
#         df_list = []
#         # Loop over each league folder
#         for league_folder in os.listdir(self.raw_data_path):
#             league_path = os.path.join(self.raw_data_path, league_folder)
#             if os.path.isdir(league_path):  # ensure it's a folder
#                 # Loop over CSV files in this folder
#                 for file in os.listdir(league_path):
#                     if file.endswith(".csv"):
#                         file_path = os.path.join(league_path, file)
#                         df = pd.read_csv(file_path)
#                         df_list.append(df)
#         return df_list

#     def merge_all_leagues(self):
#         """
#         Merge all league datasets into a single DataFrame.
#         """
#         df_list = self.load_league_files()
#         merged_df = pd.concat(df_list, ignore_index=True)
#         return merged_df

import pandas as pd
import os
import logging

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_datasets(self):
        league_name = ['ASerie','BundesLiga','LaLiga','League1','PremierLeague']
        seasons = ['1415','1516','1617','1718','1819','1920','2021','2122']
        df_list = []

        for league in league_name:
            league_folder = os.path.join(self.path, league)
            if not os.path.exists(league_folder):
                logging.warning(f"League folder not found: {league_folder}")
                continue

            for season in seasons:
                # Special case: League1 1415 has extra underscore
                if league == "League1" and season == "1415":
                    filename = f"metadata_ligue1_{season}_.csv"
                else:
                    if league == "ASerie":
                        filename = f"metadata_serie_a_{season}.csv"
                    elif league == "BundesLiga":
                        filename = f"metadata_bundesliga_{season}.csv"
                    elif league == "LaLiga":
                        filename = f"metadata_laliga_{season}.csv"
                    elif league == "League1":
                        filename = f"metadata_ligue1_{season}.csv"
                    elif league == "PremierLeague":
                        filename = f"metadata_premier_league_{season}.csv"
                    else:
                        continue

                file_path = os.path.join(league_folder, filename)

                # Load file only if it exists
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        df["league"] = league
                        df["season"] = season
                        df_list.append(df)
                        logging.info(f"Loaded file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error loading file {file_path}: {e}")
                else:
                    logging.warning(f"File not found: {file_path}, skipping.")

        # Concatenate all loaded DataFrames
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            logging.info(f"Data loaded successfully. Shape: {full_df.shape}")
            return full_df
        else:
            logging.error("No data loaded! Check your file paths and filenames.")
            return pd.DataFrame()  # return empty DataFrame if nothing loaded






