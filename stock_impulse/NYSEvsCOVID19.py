# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:34:32 2020

@author: Tharshi S and Darsh K.
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make pretty
plt.style.use('seaborn')

# import data (git pull in data folder to get newest data)
nyt_df = pd.read_csv("../data/NYT/covid-19-data/us-counties.csv")

# drop uneeded columns
nyt_df = nyt_df.drop(["county", "state", "fips"], axis=1)

# aggregate by date
nyt_df = nyt_df.groupby(['date']).sum().reset_index()

fig1 = plt.figure()
plt.semilogy(nyt_df["cases"],  ".", label="cases")
plt.semilogy(nyt_df["deaths"], ".", label="deaths")
plt.xlabel("Days since {}".format(nyt_df["date"].loc[0]))
plt.ylabel("Count")
plt.legend()
plt.title("COVID19 Cumulative # of Cases and Deaths (USA)")
plt.grid("on")
plt.savefig("cumulative_cases_deaths.pdf", dpi=600)