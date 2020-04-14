# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:02:42 2020

@author: tharshi
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EBOLA timeline: Dec 2013 - Jan 2016 = > Q4 2013 to Q1 2016 

# import IFS data
# this data is unadjusted for inflation and seasonality
# GDP is measured in domestic currency (Millions)
gdp_df = pd.read_excel("../../data/IFS/Q_nominal_GDP_Q42013toQ12016.xls",
                       header=6,
                       index_col=1,
                       na_values=["...", "-"])

# clean the data
gdp_df = gdp_df.drop("Unnamed: 0", axis=1)
gdp_df = gdp_df.dropna()

# import and clean worldbank GDP deflator data
def_df = pd.read_csv("../../data/IFS/Y_GDPDeflator_Q42013toQ12016.csv",
                       header=0,
                       index_col=0,
                       na_values=["...", "-"])
def_df = def_df.dropna()

# import and clean exchange rate data (local currency to USD)
xrate_df = pd.read_excel("../../data/IFS/Q_XRates_Q42013toQ12016.xls",
                       header=6,
                       index_col=1,
                       na_values=["...", "-"])

xrate_df = xrate_df.drop(["Unnamed: 0",
                          "Scale",
                          "Base Year"], axis=1)
xrate_df = xrate_df.dropna()

# intersect data
idx = (gdp_df.index.intersection(def_df.index)).intersection(xrate_df.index)
gdp_df = gdp_df.loc[idx]
def_df = def_df.loc[idx]
xrate_df = xrate_df.loc[idx]

# change to USD
data = gdp_df / xrate_df

# change to Q1 2016 USD
def_df = (def_df + 100)/100
def_df = def_df.div(def_df["2016Q1"], axis=0)

for c in range(data.count(1)[0]):
    col = data.columns[c]
    data[col] = data[col].div(def_df[col], axis=0)
    
# add world gdp and remove calculated categories
data.loc['World']= data.sum(numeric_only=True, axis=0)
data = data.drop("Euro Area", axis=0)

# calculate correlation matrix for top N nations
N = 15
topN_idx = data.sum(axis=1).sort_values(ascending=False).head(N).index
corr_mat = data.loc[topN_idx].T.corr()

# plot and save matrix
fig, ax = plt.subplots(figsize=(10, 10))

# tick magic
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))
ax.set_xticklabels(topN_idx)
ax.set_yticklabels(topN_idx)
if N > 30:
    f_size = 5
else:
    f_size = None

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=f_size)
plt.setp(ax.get_yticklabels(), fontsize=f_size)

# plot correlation as heatmap
im = ax.imshow(corr_mat, cmap="coolwarm", aspect="auto",vmin=-1, vmax=1)
ax.set_title("Correlation Matrix - Ebola - Top {} Contributors".format(N))
min_val = corr_mat.min().min()
max_val = corr_mat.max().max()
if N > 22:
    fig.colorbar(im)
elif 0 < N <= 22:
    for i in range(N):
        for j in range(N):
            value = round(corr_mat.iat[i, j], 2)
            if abs(value) < 0.5:
                colour="k"
            else:
                colour="w"
            text = ax.text(j, i, value, ha="center", va="center", color=colour)
else:
    pass
plt.savefig("gdp_corr_Ebola_top{}.pdf".format(N), dpi=600, bbox_inches="tight")