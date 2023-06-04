import pandas as pd
import numpy as np
# Importing Data
gdp_df_dresden = pd.read_csv('data/uncleaned/dresden/bip_dresden.csv', encoding="utf_8", encoding_errors="replace", sep=";")


#switching x and y
gdp_df_dresden = gdp_df_dresden.transpose()

#fix naming of columns
gdp_df_dresden.columns = gdp_df_dresden.iloc[0]

# Feature Extraction
gdp_per_inhabit = gdp_df_dresden.iloc[[0,2], : ].copy()
gdp_per_inhabit = gdp_per_inhabit.drop('Jahr')

all_features = [gdp_per_inhabit]

# Data Cleaning
columns_to_drop = [2000, 2001, 2021, 2022, "Jahr"]

for df in all_features:
    columns_present = set(df.columns).intersection(columns_to_drop)
    if columns_present:
        df.drop(columns=columns_present, inplace=True)

# Building the Features Dataframe
features_df = pd.concat(all_features)
features_df.reset_index(drop=True, inplace=True)
features_df = features_df.transpose()
features_df.iloc[:, 2:4] = features_df.iloc[:, 2:4].replace(',', '', regex=True).astype(float)


