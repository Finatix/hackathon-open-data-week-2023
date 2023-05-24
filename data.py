import pandas as pd

# Importing Data
gdp_df = pd.read_csv('./data/Wirtschaft_Bruttoinlandsprodukt und Bruttowertschöpfung.csv')
unemployed_df = pd.read_csv('./data/Erwerbstätigkeit und Arbeitsmarkt_Arbeitslose.csv')
age_dist_df = pd.read_csv('./data/Bevölkerungsbestand_Einwohner nach Alter.csv')
housing_stock_df = pd.read_csv('./data/Bautätigkeit und Wohnen_Wohnungsbestand.csv')
resident_df = pd.read_csv('./data/Bevölkerungsbestand_Einwohner.csv')
total_births_df = pd.read_json('./data/Geburten.json', orient='index').transpose().astype(str)
total_births_df.columns = total_births_df.columns.astype(str)

# Feature Extraction
gdp_per_inhabit = gdp_df.iloc[1:2, :].copy()
all_unemployed = unemployed_df.iloc[73:74, :].copy()
avg_age = age_dist_df.iloc[33:34, :].copy()
living_space_per_inhabit = housing_stock_df.iloc[4:5, :].copy()
total_population = resident_df.iloc[0:1, :].copy()
total_childbearing_women = age_dist_df.iloc[28:29, :].copy()

all_features = [gdp_per_inhabit, all_unemployed, avg_age, living_space_per_inhabit, total_population,
                total_childbearing_women, total_births_df]

# Data Cleaning
columns_to_drop = ['2000', '2001', '2021', '2022', 'Einheit', 'Zusammengefasste Geburtenziffer ', 'Kennziffer',
                   'Gebiet', 'Sachmerkmal']

for df in all_features:
    columns_present = set(df.columns).intersection(columns_to_drop)
    if columns_present:
        df.drop(columns=columns_present, inplace=True)

# Building the Features Dataframe
features_df = pd.concat(all_features)
features_df.reset_index(drop=True, inplace=True)
features_df = features_df.transpose()
features_df.iloc[:, 2:4] = features_df.iloc[:, 2:4].replace(',', '', regex=True).astype(float)