# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:05:58 2020

@author: tharshi
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SARS timeline: November 2002 - June 2004 = > Q4 2002 to Q2 2004 

# import IFS data
# this data is unadjusted for inflation and seasonality
# GDP is measured in domestic currency (Millions)
gdp_df = pd.read_excel("../../data/IFS/Q_nominal_GDP_2002to2004.xls",
                       header=6,
                       index_col=1,
                       na_values=["...", "-"])

# clean the data
gdp_df = gdp_df.drop("Unnamed: 0", axis=1)
gdp_df = gdp_df.dropna()

# import and clean worldbank GDP deflator data
def_df = pd.read_csv("../../data/IFS/Y_GDPDeflator_2002to2004.csv",
                       header=0,
                       index_col=0,
                       na_values=["...", "-"])
def_df = def_df.dropna()

# import and clean exchange rate data (local currency to USD)
xrate_df = pd.read_excel("../../data/IFS/Q_XRates_2002to2004.xls",
                       header=6,
                       index_col=1,
                       na_values=["...", "-"])

xrate_df = xrate_df.drop(["Unnamed: 0",
                          "Scale",
                          "Base Year"], axis=1)
# intersect data
idx = (gdp_df.index.intersection(def_df.index)).intersection(xrate_df.index)
gdp_df = gdp_df.loc[idx]
def_df = def_df.loc[idx]
xrate_df = xrate_df.loc[idx]

# change to USD
data = gdp_df / xrate_df
# change to 2004 USD
def_df = (def_df + 100)/100
def_df = def_df.div(def_df["2004"], axis=0)

# # useful lists for processing data
years = ["2002", "2003", "2004"]

for c in range(data.count(1)[0]):
    col = data.columns[c]
    for y in years:
        if y in col:
            data[col] = data[col].div(def_df[y], axis=0)
        else:
            pass

# calculate correlation matrix
data = data.T
corr_mat = data.corr()

# plot and save matrix
fig, ax = plt.subplots()

# tick magic
ax.set_xticks(np.arange(len(idx)))
ax.set_yticks(np.arange(len(idx)))
ax.set_xticklabels(idx)
ax.set_yticklabels(idx)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="5")
plt.setp(ax.get_yticklabels(), fontsize="5")

# plot correlation as heatmap
im = ax.imshow(corr_mat, cmap="coolwarm")
ax.set_title("Correlation Matrix")
fig.colorbar(im)
plt.savefig("gdp_corr_SARS.pdf", dpi=600)