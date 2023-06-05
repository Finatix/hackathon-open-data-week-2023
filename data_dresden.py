import pandas as pd
import numpy as np
# Importing Data
gdp_df = pd.read_csv('data/dresden/bip_dresden.csv', encoding="utf_8", encoding_errors="replace", sep=";")
unemployed_df = pd.read_csv('data/dresden/arbeitslosigkeit/dresden_arbeitslose_2011_2022.csv', encoding="utf_8", encoding_errors="replace", sep=",")
age_dist_df = pd.read_csv('data/dresden/alter/durchschnittsalter_2002-2020.csv', encoding="utf_8", encoding_errors="replace", sep=",")
resident_df = pd.read_csv('data/dresden/einwohner/einwohner_dresden_1999_2022.csv', encoding="utf_8", encoding_errors="replace", sep=",")

# gdp_df = pd.read_csv('./data/Wirtschaft_Bruttoinlandsprodukt und Bruttowertschöpfung.csv')
# unemployed_df = pd.read_csv('./data/Erwerbstätigkeit und Arbeitsmarkt_Arbeitslose.csv')
# age_dist_df = pd.read_csv('./data/Bevölkerungsbestand_Einwohner nach Alter.csv')
# housing_stock_df = pd.read_csv('./data/Bautätigkeit und Wohnen_Wohnungsbestand.csv')
# resident_df = pd.read_csv('./data/Bevölkerungsbestand_Einwohner.csv')
# total_births_df = pd.read_json('./data/Geburten.json', orient='index').transpose().astype(str)
# total_births_df.columns = total_births_df.columns.astype(str)


#switching x and y
gdp_df = gdp_df.transpose()

#fix naming of columns
gdp_df.columns = gdp_df.iloc[0]

#fix datatype
all_dfs = [gdp_df, unemployed_df, resident_df, age_dist_df]
for df in all_dfs:
    df = df.astype(float, errors="ignore")

# Feature Extraction
gdp_per_inhabit = gdp_df.iloc[[0, 2], :].copy()
gdp_per_inhabit = gdp_per_inhabit.drop('Jahr')
total_population = resident_df
all_unemployed = unemployed_df
avg_age = age_dist_df

all_features = [gdp_per_inhabit, all_unemployed, avg_age, total_population]


#gdp_per_inhabit, all_unemployed, avg_age, living_space_per_inhabit, total_population,
                #total_childbearing_women, total_births_df

# Data Cleaning


columns_to_drop = ['', 1999, 2000, 2001, 2021, 2022, "Jahr", "1999", "2000", "2001", "2021", "2022", "Jahr "]

years = pd.read_csv("data/years_correct_format.csv", encoding="utf_8", encoding_errors="replace", sep=",")
for df in all_features:
    columns_present = set(df.columns).intersection(columns_to_drop)
    if columns_present:
        df.drop(columns=columns_present, inplace=True)
    df.columns = years.columns #fixes formatting of column labels

# Building the Features Dataframe
features_df = pd.concat(all_features, ignore_index=True)
features_df.reset_index(drop=True, inplace=True)
features_df = features_df.transpose()
features_df.iloc[:, 2:4] = features_df.iloc[:, 2:4].replace(',', '.', regex=True).astype(float)


