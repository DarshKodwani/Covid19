# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:11:07 2020

@author: tharshi
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import raw world bank data
# Country Name # Country Code # Year # Value #
bank_df = pd.read_csv("../../data/datahub/gdp_csv.txt")

# useful lists for extracting data
yr_list = list(np.arange(1960, 2016 + 1))
u_countries = bank_df["Country Name"].unique()
countries = bank_df["Country Name"].unique().tolist()

# keep countries with records from 1960 - 2016
for country in u_countries:
    val_list = bank_df[bank_df["Country Name"] == country]["Year"].tolist()
    if not all(y in val_list for y in yr_list):
        bank_df = bank_df[bank_df["Country Name"] != country]
        countries.remove(country)
    else:
        pass
    
# create new dataframe with select countries as columns
data = bank_df.pivot(index="Year", columns="Country Name", values="Value")

# calculate covariance and correlation matrices
corr_mat = data.corr()

# plot and save matrix
fig, ax = plt.subplots()

# tick magic
ax.set_xticks(np.arange(len(countries)))
ax.set_yticks(np.arange(len(countries)))
ax.set_xticklabels(countries)
ax.set_yticklabels(countries)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="1")
plt.setp(ax.get_yticklabels(), fontsize="2")

im = ax.imshow(corr_mat, cmap="binary")
ax.set_title("Correlation Matrix")
fig.colorbar(im)
plt.savefig("gdp_corr_1960to2016.pdf", dpi=600)

# remove derived quantities and redo correlation calculation
dropped_countries = ["Caribbean small states",
                  "Early-demographic dividend", 
                  "East Asia & Pacific", 
                  "East Asia & Pacific (excluding high income)",
                  "East Asia & Pacific (IDA & IBRD countries)",
                  "Euro area",
                  "Europe & Central Asia",
                  "European Union",
                  "Heavily indebted poor countries (HIPC)",
                  "High income",
                  "IBRD only",
                  "IDA & IBRD total",
                  "IDA blend",
                  "IDA total",
                  "Late-demographic dividend",
                  "Latin America & Caribbean",
                  "Latin America & Caribbean (excluding high income)",
                  "Latin America & the Caribbean (IDA & IBRD countries)",
                  "Low & middle income",
                  "Middle income",
                  "North America",
                  "OECD members",
                  "Post-demographic dividend",
                  "Pre-demographic dividend",
                  "South Asia",
                  "South Asia (IDA & IBRD)",
                  "Sub-Saharan Africa",
                  "Sub-Saharan Africa (excluding high income)",
                  "Sub-Saharan Africa (IDA & IBRD countries)",
                  "Upper middle income"]

data_concise = data.drop(dropped_countries, axis=1)
countries_concise = [c for c in countries if c not in dropped_countries]

# calculate covariance and correlation matrices
corr_mat_concise = data_concise.corr()

# plot and save matrix
fig2, ax2 = plt.subplots()

# tick magic
ax2.set_xticks(np.arange(len(countries_concise)))
ax2.set_yticks(np.arange(len(countries_concise)))
ax2.set_xticklabels(countries_concise)
ax2.set_yticklabels(countries_concise)
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="2")
plt.setp(ax2.get_yticklabels(), fontsize="3")

im = ax2.imshow(corr_mat_concise, cmap="binary")
ax2.set_title("Correlation Matrix (Concise)")
fig2.colorbar(im)
plt.savefig("gdp_corr_1960to2016_concise.pdf", dpi=600)

# focus on top economies and the world
important_countries = ["United States", "China", "Japan", "United Kingdom",
                       "France", "India", "Italy", "Canada", "World"]

data_top = data[important_countries]
countries_top = [c for c in countries if c in important_countries]

# calculate covariance and correlation matrices
corr_mat_top = data_top.corr()

# plot and save matrix
fig3, ax3 = plt.subplots()

n_countries_top = len(countries_top)

# tick magic
ax3.set_xticks(np.arange(n_countries_top))
ax3.set_yticks(np.arange(n_countries_top))
ax3.set_xticklabels(countries_top)
ax3.set_yticklabels(countries_top)
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize="5")
plt.setp(ax3.get_yticklabels(), fontsize="7")

# label values
for i in range(n_countries_top):
    for j in range(n_countries_top):
        value = round(corr_mat_top.iat[i, j], 2)
        if value < 0.65:
            colour="k"
        else:
            colour="w"
        text = ax3.text(j, i, value, ha="center", va="center", color=colour)

im = ax3.imshow(corr_mat_top, cmap="binary")
ax3.set_title("Correlation Matrix for Top Countries")
plt.savefig("gdp_corr_1960to2016_top.pdf", dpi=600)